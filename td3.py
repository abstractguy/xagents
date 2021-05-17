import tensorflow as tf

from base_agents import OffPolicy


class TD3(OffPolicy):
    def __init__(self, envs, actor_model, critic_model, **kwargs):
        super(TD3, self).__init__(envs, actor_model, **kwargs)
        self.actor = self.model
        self.critic = critic_model
        self.target_actor = tf.keras.models.clone_model(self.actor)
        self.target_critic = tf.keras.models.clone_model(self.critic)

    def train_step(self):
        pass


if __name__ == '__main__':
    from utils import ModelHandler, create_gym_env

    seed = 55
    en = create_gym_env('BipedalWalker-v3', 16, False)
