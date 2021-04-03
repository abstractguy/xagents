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
