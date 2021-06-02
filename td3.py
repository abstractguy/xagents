import tensorflow as tf
from tensorflow.keras.losses import MSE

from base_agents import OffPolicy


class TD3(OffPolicy):
    def __init__(
        self,
        envs,
        actor_model,
        critic_model,
        buffers,
        policy_delay=2,
        gradient_steps=None,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        **kwargs,
    ):
        super(TD3, self).__init__(envs, actor_model, buffers, **kwargs)
        self.actor = actor_model
        self.critic1 = critic_model
        self.policy_delay = policy_delay
        self.gradient_steps = gradient_steps
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.episode_steps = tf.Variable(tf.zeros(self.n_envs), False)
        self.step_increment = tf.ones(self.n_envs)
        self.critic2 = tf.keras.models.clone_model(self.critic1)
        self.critic2.compile(
            tf.keras.optimizers.get(self.critic1.optimizer.get_config()['name'])
        )
        self.critic2.optimizer.learning_rate.assign(
            self.critic1.optimizer.learning_rate
        )
        self.target_actor = tf.keras.models.clone_model(self.actor)
        self.target_critic1 = tf.keras.models.clone_model(self.critic1)
        self.target_critic2 = tf.keras.models.clone_model(self.critic1)
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic1.set_weights(self.critic1.get_weights())
        self.target_critic2.set_weights(self.critic2.get_weights())
        self.model_groups = [
            (self.actor, self.target_actor),
            (self.critic1, self.target_critic1),
            (self.critic2, self.target_critic2),
        ]
        self.batch_dtypes = 5 * [tf.float32]

    def update_target_models(self):
        for model, target_model in self.model_groups:
            for var, target_var in zip(
                model.trainable_variables, target_model.trainable_variables
            ):
                target_var.assign((1 - self.tau) * target_var + self.tau * var)

    def update_critic_weights(self, states, actions, new_states, dones, rewards):
        with tf.GradientTape(True) as tape:
            noise = (
                tf.random.normal(
                    (self.buffers[0].batch_size * self.n_envs, self.n_actions)
                )
                * self.policy_noise
            )
            noise = tf.clip_by_value(noise, -self.noise_clip, self.noise_clip)
            new_actions = tf.clip_by_value(
                self.target_actor(new_states) + noise, -1.0, 1.0
            )
            target_critic_input = tf.concat([new_states, new_actions], 1)
            target_value1 = self.target_critic1(target_critic_input)
            target_value2 = self.target_critic2(target_critic_input)
            target_value = tf.minimum(target_value1, target_value2)
            target_value = rewards + tf.stop_gradient(
                (1 - dones) * self.gamma * target_value
            )
            critic_input = tf.concat([states, actions], 1)
            value1 = self.critic1(critic_input)
            value2 = self.critic2(critic_input)
            critic1_loss, critic2_loss = MSE(value1, target_value), MSE(
                value2, target_value
            )
        self.critic1.optimizer.minimize(
            critic1_loss, self.critic1.trainable_variables, tape=tape
        )
        self.critic2.optimizer.minimize(
            critic2_loss, self.critic2.trainable_variables, tape=tape
        )

    def update_actor_weights(self, states):
        with tf.GradientTape() as tape:
            actor_loss = -tf.reduce_mean(
                self.critic1(tf.concat([states, self.actor(states)], 1))
            )
        self.actor.optimizer.minimize(
            actor_loss, self.actor.trainable_variables, tape=tape
        )

    def update_weights(self, gradient_steps):
        for gradient_step in range(int(gradient_steps)):
            states, actions, rewards, dones, new_states = tf.numpy_function(
                self.concat_buffer_samples, [], self.batch_dtypes
            )
            self.update_critic_weights(states, actions, new_states, dones, rewards)
            if gradient_step % self.policy_delay == 0:
                self.update_actor_weights(states)
                self.update_target_models()

    @tf.function
    def train_step(self):
        step_actions = self.actor(tf.numpy_function(self.get_states, [], tf.float32))
        *_, dones, _ = tf.numpy_function(
            self.step_envs, [step_actions, True, True], self.batch_dtypes
        )
        for done_idx in tf.where(dones):
            gradient_steps = self.gradient_steps or self.episode_steps[done_idx[0]]
            self.update_weights(gradient_steps)
        self.episode_steps.assign(
            (self.episode_steps + self.step_increment) * (1 - dones)
        )


if __name__ == '__main__':
    from utils import (IAmTheOtherKindOfReplayBufferBecauseFuckTensorflow,
                       ModelReader, create_gym_env)

    en = create_gym_env('BipedalWalker-v3', 16, False)
    amr = ModelReader(
        'models/ann/td3-actor.cfg',
        [en[0].action_space.shape[0]],
        en[0].observation_space.shape,
        'adam',
    )
    cmr = ModelReader(
        'models/ann/td3-critic.cfg',
        [1],
        en[0].observation_space.shape[0] + en[0].action_space.shape[0],
        'adam',
    )
    am = amr.build_model()
    cm = cmr.build_model()
    bs = [
        IAmTheOtherKindOfReplayBufferBecauseFuckTensorflow(
            1000000 // len(en),
            5,
            initial_size=100 // len(en),
            batch_size=100 // len(en),
        )
        for _ in range(len(en))
    ]
    agent = TD3(en, am, cm, bs)
    agent.fit(250)
