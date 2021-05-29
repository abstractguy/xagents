import tensorflow as tf

from base_agents import OffPolicy


class TD3(OffPolicy):
    def __init__(self, envs, actor_model, critic_model, **kwargs):
        super(TD3, self).__init__(envs, actor_model, **kwargs)
        self.actor = actor_model
        self.critic1 = critic_model
        self.critic2 = tf.keras.models.clone_model(critic_model)
        self.target_actor = tf.keras.models.clone_model(actor_model)
        self.target_critic1 = tf.keras.models.clone_model(critic_model)
        self.target_critic2 = tf.keras.models.clone_model(critic_model)


if __name__ == '__main__':
    from utils import ModelReader, create_gym_env

    en = create_gym_env('BipedalWalker-v3', 16, False)
    amh = ModelReader(
        'models/ann/td3-actor.cfg',
        [en[0].action_space.shape[0]],
        en[0].observation_space.shape,
        'adam',
    )
    cmh = ModelReader(
        'models/ann/td3-critic.cfg',
        [1],
        en[0].observation_space.shape[0] + en[0].action_space.shape[0],
        'adam',
    )
    am = amh.build_model()
    cm = cmh.build_model()
    agn = TD3(
        en,
        am,
        cm,
        buffer_max_size=1000000,
        buffer_initial_size=25000,
        buffer_batch_size=128,
    )
