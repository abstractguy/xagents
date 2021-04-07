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

    # @tf.function
    def train_step(self):
        states, actions, returns, values, log_probs = tf.numpy_function(
            self.get_batch, [], 5 * [tf.float32]
        )


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
