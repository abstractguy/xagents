import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions import Categorical

from ppo import PPO


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
        actor_iterations=10,
        critic_iterations=3,
        critic_learning_rate=3e-4,
        fvp_n_steps=5,
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
        self.actor_iterations = actor_iterations
        self.critic_learning_rate = critic_learning_rate
        self.fvp_n_steps = fvp_n_steps

    @staticmethod
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

    @staticmethod
    def weights_to_flat(to_flatten, trainable_variables=None):
        if not trainable_variables:
            to_concat = [tf.reshape(non_flat, [-1]) for non_flat in to_flatten]
        else:
            to_concat = [
                tf.reshape(
                    non_flat
                    if non_flat is not None
                    else tf.zeros_like(trainable_variable),
                    [-1],
                )
                for (non_flat, trainable_variable) in zip(
                    to_flatten, trainable_variables
                )
            ]
        return tf.concat(to_concat, 0)

    def calculate_fvp(self, flat_tangent, states):
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                old_actor_output = self.get_model_outputs(
                    states, [self.old_actor, self.critic]
                )[4]
                new_actor_output = self.get_model_outputs(
                    states, [self.actor, self.critic]
                )[4]
                old_distribution = Categorical(old_actor_output)
                new_distribution = Categorical(new_actor_output)
                kl_divergence = old_distribution.kl_divergence(new_distribution)
                mean_kl = tf.reduce_mean(kl_divergence)
            kl_grads = tape1.gradient(mean_kl, self.actor.trainable_variables)
            tangents = self.flat_to_weights(
                flat_tangent, self.actor.trainable_variables
            )
            gvp = tf.add_n(
                [
                    tf.reduce_sum(grad * tangent)
                    for (grad, tangent) in zip(kl_grads, tangents)
                ]
            )
        hessians_products = tape2.gradient(gvp, self.actor.trainable_variables)
        return (
            self.weights_to_flat(hessians_products, self.actor.trainable_variables)
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

    def optimizer_step(self, flat_update):
        self.critic_updates += 1
        a = (
            self.critic_learning_rate
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
        update = self.weights_to_flat(self.critic.trainable_variables) + step
        self.flat_to_weights(update, self.critic.trainable_variables, True)

    def calculate_kl_divergence(self, states):
        old_actor_output = self.get_model_outputs(
            states, [self.old_actor, self.critic]
        )[4]
        new_actor_output = self.get_model_outputs(states, [self.actor, self.critic])[4]
        old_distribution = Categorical(old_actor_output)
        new_distribution = Categorical(new_actor_output)
        return (
            old_distribution.kl_divergence(new_distribution),
            old_distribution,
            new_distribution,
        )

    def calculate_losses(self, states, actions, advantages):
        (
            kl_divergence,
            old_distribution,
            new_distribution,
        ) = self.calculate_kl_divergence(states)
        kl_divergence = tf.reduce_mean(old_distribution.kl_divergence(new_distribution))
        entropy = tf.reduce_mean(new_distribution.entropy())
        entropy_loss = self.entropy_coef * entropy
        ratio = tf.exp(
            new_distribution.log_prob(actions) - old_distribution.log_prob(actions)
        )
        surrogate_gain = tf.reduce_mean(ratio * advantages)
        surrogate_loss = surrogate_gain + entropy_loss
        return [surrogate_loss, kl_divergence, entropy_loss, surrogate_gain, entropy]

    def at_step_start(self):
        self.old_actor.set_weights(self.actor.get_weights())

    def update_actor_weights(
        self,
        flat_weights,
        full_step,
        surrogate_loss,
        states,
        actions,
        advantages,
        expected_improvement,
    ):
        step_size = 1.0
        for _ in range(self.actor_iterations):
            updated_weights = flat_weights + full_step * step_size
            self.flat_to_weights(updated_weights, self.actor.trainable_variables, True)
            losses = new_surrogate_loss, new_kl_divergence, *_ = self.calculate_losses(
                states, actions, advantages
            )
            improvement = new_surrogate_loss - surrogate_loss
            print(f'Expected: {expected_improvement} Got: {improvement}')
            if not np.isfinite(losses).all():
                print('Got non-finite value of losses -- bad!')
            elif new_kl_divergence > self.max_kl * 1.5:
                print('Violated KL constraint. shrinking step.')
            elif improvement < 0:
                print('Surrogate did not improve. shrinking step.')
            else:
                print('Step size OK!')
                break
            step_size *= 0.5
        else:
            print('Could not compute a good step')
            self.flat_to_weights(flat_weights, self.actor.trainable_variables, True)

    def update_critic_weights(self, states, returns):
        for _ in range(self.critic_iterations):
            for (states_mb, returns_mb) in self.get_mini_batches(states, returns):
                with tf.GradientTape() as tape:
                    values = self.get_model_outputs(
                        states_mb, [self.actor, self.critic]
                    )[2]
                    value_loss = tf.reduce_mean(tf.square(values - returns_mb))
                grads = self.weights_to_flat(
                    tape.gradient(value_loss, self.critic.trainable_variables),
                    self.critic.trainable_variables,
                )
                self.optimizer_step(grads)

    @tf.function
    def train_step(self):
        states, actions, returns, values, _ = tf.numpy_function(
            self.get_batch, [], 5 * [tf.float32]
        )
        advantages = returns - values
        advantages = (advantages - tf.reduce_mean(advantages)) / tf.math.reduce_std(
            advantages
        )
        with tf.GradientTape() as tape:
            losses = self.calculate_losses(states, actions, advantages)
        flat_grads = self.weights_to_flat(
            tape.gradient(losses[0], self.actor.trainable_variables),
            self.actor.trainable_variables,
        )
        if not tf.numpy_function(np.allclose, [flat_grads, 0], tf.bool):
            pass
        step_direction = self.conjugate_gradients(
            flat_grads, states[:: self.fvp_n_steps]
        )
        if not tf.reduce_all(tf.math.is_finite(step_direction)):
            pass
        shs = 0.5 * tf.tensordot(
            step_direction,
            self.calculate_fvp(step_direction, states[:: self.fvp_n_steps]),
            1,
        )
        lagrange_multiplier = tf.math.sqrt(shs / self.max_kl)
        full_step = step_direction / lagrange_multiplier
        expected_improvement = tf.tensordot(flat_grads, full_step, 1)
        surrogate_loss = losses[0]
        pre_actor_weights = self.weights_to_flat(
            self.actor.trainable_variables,
        )
        tf.numpy_function(
            self.update_actor_weights,
            [
                pre_actor_weights,
                full_step,
                surrogate_loss,
                states,
                actions,
                advantages,
                expected_improvement,
            ],
            [],
        )
        self.update_critic_weights(states, returns)


if __name__ == '__main__':
    from utils import ModelHandler, create_gym_env

    en = create_gym_env('PongNoFrameskip-v4', 16)
    a_mh = ModelHandler('models/cnn-a-tiny.cfg', [en[0].action_space.n])
    c_mh = ModelHandler('models/cnn-c-tiny.cfg', [1])
    a_m = a_mh.build_model()
    c_m = c_mh.build_model()
    agn = TRPO(en, a_m, c_m)
    agn.fit(19)
