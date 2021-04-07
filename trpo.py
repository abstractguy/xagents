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
        n_steps=512,
        lam=1,
        fvp_n_steps=5,
        entropy_coef=0,
        cg_iterations=10,
        cg_residual_tolerance=1e-10,
        cg_damping=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        *args,
        **kwargs,
    ):
        super(TRPO, self).__init__(
            envs, actor_model, n_steps, lam, entropy_coef=entropy_coef, *args, **kwargs
        )
        self.actor = self.model
        self.critic = critic_model
        self.output_models = self.actor, self.critic
        self.old_actor = tf.keras.models.clone_model(self.actor)
        self.old_critic = tf.keras.models.clone_model(self.critic)
        self.fvp_n_steps = fvp_n_steps
        self.cg_iterations = cg_iterations
        self.cg_residual_tolerance = cg_residual_tolerance
        self.cg_damping = cg_damping
        self.critic_updates = 0
        self.critic_flat_size = sum(
            tf.math.reduce_prod(v.shape) for v in self.critic.trainable_variables
        )
        self.optimizer_beta1 = beta1
        self.optimizer_beta2 = beta2
        self.optimizer_epsilon = epsilon
        self.optimizer_m = tf.zeros(self.critic_flat_size)
        self.optimizer_v = tf.zeros(self.critic_flat_size)

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
        update = self.weights_to_flat(self.critic.trainable_variables) + step
        self.flat_to_weights(update, self.critic.trainable_variables, True)

    def at_step_start(self):
        self.old_actor.set_weights(self.actor.get_weights())

    def calculate_kl_divergence(self, states):
        *_, old_actor_output = self.get_model_outputs(
            states, [self.old_actor, self.critic]
        )
        *_, new_actor_output = self.get_model_outputs(states, [self.actor, self.critic])
        old_distribution = Categorical(old_actor_output)
        new_distribution = Categorical(new_actor_output)
        return (
            tf.reduce_mean(old_distribution.kl_divergence(new_distribution)),
            old_distribution,
            new_distribution,
        )

    def calculate_losses(self, states, actions, returns, values):
        advantages = tf.stop_gradient(returns - values)
        advantages = (advantages - tf.reduce_mean(advantages)) / tf.math.reduce_std(
            advantages
        )
        (
            kl_divergence,
            old_distribution,
            new_distribution,
        ) = self.calculate_kl_divergence(states)
        entropy = tf.reduce_mean(new_distribution.entropy()) * self.entropy_coef
        ratio = tf.exp(
            new_distribution.log_prob(actions) - old_distribution.log_prob(actions)
        )
        surrogate_loss = tf.reduce_mean(ratio * advantages) + entropy
        return surrogate_loss, kl_divergence

    def calculate_fvp(self, flat_tangent, states):
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                kl_divergence, *_ = self.calculate_kl_divergence(states)
            kl_grads = tape1.gradient(kl_divergence, self.actor.trainable_variables)
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

    @tf.function
    def train_step(self):
        states, actions, returns, values, log_probs = tf.numpy_function(
            self.get_batch, [], 5 * [tf.float32]
        )
        with tf.GradientTape() as tape:
            surrogate_loss, kl_divergence = self.calculate_losses(
                states, actions, returns, values
            )
        grads = tape.gradient(surrogate_loss, self.actor.trainable_variables)
        grads = self.weights_to_flat(grads, self.actor.trainable_variables)
        if tf.numpy_function(np.allclose, [grads, 0], tf.bool):
            pass
        step_direction = self.conjugate_gradients(grads, states[:: self.fvp_n_steps])


if __name__ == '__main__':
    from utils import ModelHandler, create_gym_env

    sd = None

    envi = create_gym_env('PongNoFrameskip-v4', 2)
    actor_mh = ModelHandler('models/cnn-a-tiny.cfg', [envi[0].action_space.n])
    critic_mh = ModelHandler('models/cnn-c-tiny.cfg', [1])
    actor_m = actor_mh.build_model()
    critic_m = critic_mh.build_model()
    agn = TRPO(envi, actor_m, critic_m)
    agn.fit(19)
