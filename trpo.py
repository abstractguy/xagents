import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions import Categorical

from ppo import PPO


def flat_to_weights(flat, trainable_variables, in_place=False):
    updated_trainable_variables = []
    start_idx = 0
    for trainable_variable in trainable_variables:
        shape = trainable_variable.shape
        flat_size = tf.math.reduce_prod(shape)
        updated_trainable_variable = tf.reshape(
            flat[start_idx : start_idx + flat_size], shape
        )
        if in_place:
            trainable_variable.assign(updated_trainable_variable)
        else:
            updated_trainable_variables.append(updated_trainable_variable)
        start_idx += flat_size
    return updated_trainable_variables


def weights_to_flat(to_flatten, trainable_variables=None):
    if not trainable_variables:
        to_concat = [tf.reshape(non_flat, [-1]) for non_flat in to_flatten]
    else:
        to_concat = [
            tf.reshape(
                non_flat if non_flat is not None else tf.zeros_like(trainable_variable),
                [-1],
            )
            for (non_flat, trainable_variable) in zip(to_flatten, trainable_variables)
        ]
    return tf.concat(to_concat, 0)


def iterbatches(
    arrays,
    *,
    num_batches=None,
    batch_size=None,
    shuffle=True,
    include_final_partial_batch=True,
):
    assert (num_batches is None) != (
        batch_size is None
    ), 'Provide num_batches or batch_size, but not both'
    arrays = tuple(map(np.asarray, arrays))
    n = arrays[0].shape[0]
    assert all(a.shape[0] == n for a in arrays[1:])
    inds = np.arange(n)
    if shuffle:
        np.random.shuffle(inds)
    sections = np.arange(0, n, batch_size)[1:] if num_batches is None else num_batches
    for batch_inds in np.array_split(inds, sections):
        if include_final_partial_batch or len(batch_inds) == batch_size:
            yield tuple(a[batch_inds] for a in arrays)


class TRPO(PPO):
    def __init__(
        self,
        envs,
        actor_model,
        critic_model,
        entropy_coef=0.0,
        lam=1.0,
        n_steps=512,
        max_kl=1e-3,
        cg_iterations=10,
        cg_residual_tolerance=1e-10,
        cg_damping=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        critic_iterations=3,
        critic_learning_rate=3e-4,
        **kwargs,
    ):
        super(TRPO, self).__init__(
            envs,
            actor_model,
            n_steps=n_steps,
            lam=lam,
            entropy_coef=entropy_coef,
            **kwargs,
        )
        self.old_actor = tf.keras.models.clone_model(self.model)
        self.actor = self.model
        self.critic = critic_model
        self.output_models = self.actor, self.critic
        self.cg_iterations = cg_iterations
        self.cg_residual_tolerance = cg_residual_tolerance
        self.cg_damping = cg_damping
        self.max_kl = max_kl
        self.critic_updates = 0
        self.critic_flat_size = sum(
            tf.math.reduce_prod(v.shape) for v in self.critic.trainable_variables
        )
        self.optimizer_beta1 = beta1
        self.optimizer_beta2 = beta2
        self.optimizer_epsilon = epsilon
        self.optimizer_m = tf.zeros(self.critic_flat_size)
        self.optimizer_v = tf.zeros(self.critic_flat_size)
        self.critic_iterations = critic_iterations
        self.critic_learning_rate = critic_learning_rate

    def calculate_fvp(self, flat_tangent, states):
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                old_actor_output = self.old_actor(states)
                new_actor_output = self.actor(states)
                old_distribution = Categorical(old_actor_output)
                new_distribution = Categorical(new_actor_output)
                kl_divergence = old_distribution.kl_divergence(new_distribution)
                mean_kl = tf.reduce_mean(kl_divergence)
            kl_grads = tape1.gradient(mean_kl, self.actor.trainable_variables)
            tangents = flat_to_weights(flat_tangent, self.actor.trainable_variables)
            gvp = tf.add_n(
                [
                    tf.reduce_sum(grad * tangent)
                    for (grad, tangent) in zip(kl_grads, tangents)
                ]
            )
        hessians_products = tape2.gradient(gvp, self.actor.trainable_variables)
        return (
            weights_to_flat(hessians_products, self.actor.trainable_variables)
            + self.cg_damping * flat_tangent
        )

    def conjugate_gradients(self, flat_grads, states):
        p = tf.identity(flat_grads)
        r = tf.identity(flat_grads)
        x = tf.zeros_like(flat_grads)
        r_dot_r = tf.tensordot(r, r, 1)
        iterations = 0
        while tf.less(iterations, self.cg_iterations) and tf.greater(
            r_dot_r, self.cg_residual_tolerance
        ):
            z = self.calculate_fvp(p, states)
            v = r_dot_r / tf.tensordot(p, z, 1)
            x += v * p
            r -= v * z
            new_r_dot_r = tf.tensordot(r, r, 1)
            mu = new_r_dot_r / r_dot_r
            p = r + mu * p
            r_dot_r = new_r_dot_r
            iterations += 1
        return x

    def update_critic_weights(self, flat_update, step_size):
        self.critic_updates += 1
        a = (
            step_size
            * tf.math.sqrt(1 - self.optimizer_beta2 ** self.critic_updates)
            / (1 - self.optimizer_beta1 ** self.critic_updates)
        )
        self.optimizer_m = (
            self.optimizer_beta1 * self.optimizer_m
            + (1 - self.optimizer_beta1) * flat_update
        )
        self.optimizer_v = self.optimizer_beta2 * self.optimizer_v + (
            1 - self.optimizer_beta2
        ) * (flat_update * flat_update)
        step = (
            (-a)
            * self.optimizer_m
            / (tf.math.sqrt(self.optimizer_v) + self.optimizer_epsilon)
        )
        update = weights_to_flat(self.critic.trainable_variables) + step
        flat_to_weights(update, self.critic.trainable_variables, True)

    def train(self):
        np.set_printoptions(precision=3)
        loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]
        self.init_training(19, None, None, None)

        @tf.function
        def compute_lossandgrad(ob, ac, atarg):
            with tf.GradientTape() as tape:
                old_actor_output = self.old_actor(ob)
                new_actor_output = self.actor(ob)
                old_distribution = Categorical(old_actor_output)
                new_distribution = Categorical(new_actor_output)
                kloldnew = old_distribution.kl_divergence(new_distribution)
                ent = new_distribution.entropy()
                meankl = tf.reduce_mean(kloldnew)
                meanent = tf.reduce_mean(ent)
                entbonus = self.entropy_coef * meanent
                ratio = tf.exp(
                    new_distribution.log_prob(ac) - old_distribution.log_prob(ac)
                )
                surrgain = tf.reduce_mean(ratio * atarg)
                optimgain = surrgain + entbonus
                losses = [optimgain, meankl, entbonus, surrgain, meanent]
            gradients = tape.gradient(optimgain, self.actor.trainable_variables)
            return losses + [weights_to_flat(gradients, self.actor.trainable_variables)]

        @tf.function
        def compute_losses(ob, ac, atarg):
            old_actor_output = self.old_actor(ob)
            new_actor_output = self.actor(ob)
            old_distribution = Categorical(old_actor_output)
            new_distribution = Categorical(new_actor_output)
            kloldnew = old_distribution.kl_divergence(new_distribution)
            ent = new_distribution.entropy()
            meankl = tf.reduce_mean(kloldnew)
            meanent = tf.reduce_mean(ent)
            entbonus = self.entropy_coef * meanent
            ratio = tf.exp(
                new_distribution.log_prob(ac) - old_distribution.log_prob(ac)
            )
            surrgain = tf.reduce_mean(ratio * atarg)
            optimgain = surrgain + entbonus
            losses = [optimgain, meankl, entbonus, surrgain, meanent]
            return losses

        @tf.function
        def compute_vflossandgrad(ob, ret):
            with tf.GradientTape() as tape:
                pi_vf = self.critic(ob)
                vferr = tf.reduce_mean(tf.square(pi_vf - ret))
            return weights_to_flat(
                tape.gradient(vferr, self.critic.trainable_variables),
                self.critic.trainable_variables,
            )

        while True:
            print(100 * '=')
            self.check_episodes()
            if self.training_done():
                break
            self.updates += 1
            (states, actions, returns, values, _) = tf.numpy_function(
                self.get_batch, [], 5 * [tf.float32]
            )
            atarg = returns - values
            atarg = (atarg - tf.reduce_mean(atarg)) / tf.math.reduce_std(atarg)
            args = states, actions, atarg
            fvpargs = [arr[::5] for arr in args]

            self.old_actor.set_weights(self.actor.get_weights())
            *lossbefore, g = compute_lossandgrad(*args)
            lossbefore = np.array(lossbefore)
            if np.allclose(g, 0):
                print("Got zero gradient. not updating")
            else:
                stepdir = self.conjugate_gradients(g, fvpargs[0])
                assert np.isfinite(stepdir).all()
                shs = 0.5 * tf.tensordot(
                    stepdir, self.calculate_fvp(stepdir, fvpargs[0]), 1
                )
                lagrange_multiplier = np.sqrt(shs / self.max_kl)
                fullstep = stepdir / lagrange_multiplier
                expectedimprove = tf.tensordot(g, fullstep, 1)
                surrbefore = lossbefore[0]
                stepsize = 1.0
                thbefore = weights_to_flat(
                    self.actor.trainable_variables,
                    self.actor.trainable_variables,
                )
                for _ in range(10):
                    thnew = thbefore + fullstep * stepsize
                    flat_to_weights(thnew, self.actor.trainable_variables, True)
                    meanlosses = surr, kl, *_ = np.array(compute_losses(*args))
                    improve = surr - surrbefore
                    print("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
                    if not np.isfinite(meanlosses).all():
                        print("Got non-finite value of losses -- bad!")
                    elif kl > self.max_kl * 1.5:
                        print("violated KL constraint. shrinking step.")
                    elif improve < 0:
                        print("surrogate didn't improve. shrinking step.")
                    else:
                        print("Stepsize OK!")
                        break
                    stepsize *= 0.5
                else:
                    print("couldn't compute a good step")
                    flat_to_weights(thbefore, self.actor.trainable_variables, True)
            for (lossname, lossval) in zip(loss_names, meanlosses):
                print(lossname, np.around(lossval, 2))
            for _ in range(self.critic_iterations):
                for (mbob, mbret) in iterbatches(
                    [states, returns],
                    include_final_partial_batch=False,
                    batch_size=64,
                ):
                    g = compute_vflossandgrad(mbob, mbret).numpy()
                    self.update_critic_weights(g, self.critic_learning_rate)


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]


if __name__ == '__main__':
    from utils import ModelHandler, create_gym_env

    en = create_gym_env('PongNoFrameskip-v4')
    a_mh = ModelHandler('models/cnn-a-tiny.cfg', [en[0].action_space.n])
    c_mh = ModelHandler('models/cnn-c-tiny.cfg', [1])
    a_m = a_mh.build_model()
    c_m = c_mh.build_model()
    agn = TRPO(en, a_m, c_m)
    agn.train()
