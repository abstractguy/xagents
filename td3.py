import numpy as np
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
        self.episode_steps = np.zeros(self.n_envs)
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

    def critic_loss(self, obs, action, next_obs, done, reward):
        noise = tf.random.normal(action.shape) * self.policy_noise
        noise = tf.clip_by_value(noise, -self.noise_clip, self.noise_clip)
        next_action = tf.clip_by_value(self.target_actor(next_obs) + noise, -1.0, 1.0)
        target_critic_input = tf.concat([next_obs, next_action], 1)
        target_q1 = self.target_critic1(target_critic_input)
        target_q2 = self.target_critic2(target_critic_input)
        target_q = tf.minimum(target_q1, target_q2)
        target_q = reward + tf.stop_gradient((1 - done) * self.gamma * target_q)
        critic_input = tf.concat([obs, action], 1)
        current_q1 = self.critic1(critic_input)
        current_q2 = self.critic2(critic_input)
        return MSE(current_q1, target_q), MSE(current_q2, target_q)

    @tf.function
    def actor_loss(self, obs):
        return -tf.reduce_mean(self.critic1(tf.concat([obs, self.actor(obs)], 1)))

    @tf.function
    def update_targets(self):
        for model, target_model in self.model_groups:
            for var, target_var in zip(
                model.trainable_variables, target_model.trainable_variables
            ):
                target_var.assign((1 - self.tau) * target_var + self.tau * var)

    @tf.function
    def _train_critic(self, obs, action, next_obs, done, reward):
        with tf.GradientTape(True) as tape:
            critic1_loss, critic2_loss = self.critic_loss(
                obs, action, next_obs, done, reward
            )
        critic1_grads = tape.gradient(critic1_loss, self.critic1.trainable_variables)
        critic2_grads = tape.gradient(critic2_loss, self.critic2.trainable_variables)
        self.critic1.optimizer.apply_gradients(
            zip(critic1_grads, self.critic1.trainable_variables)
        )
        self.critic2.optimizer.apply_gradients(
            zip(critic2_grads, self.critic2.trainable_variables)
        )

    @tf.function
    def _train_actor(self, obs):
        with tf.GradientTape() as actor_tape:
            actor_tape.watch(self.actor.trainable_variables)
            actor_loss = self.actor_loss(obs)
        grads_actor = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(
            zip(grads_actor, self.actor.trainable_variables)
        )

    def train(self, gradient_steps):
        for gradient_step in range(gradient_steps):
            states, actions, rewards, dones, new_states = self.concat_buffer_samples()
            self._train_critic(states, actions, new_states, dones, rewards)
            if gradient_step % self.policy_delay == 0:
                self._train_actor(states)
                self.update_targets()

    def train_step(self):
        step_actions = self.actor(self.get_states())
        *_, dones, _ = self.step_envs(step_actions, True, True)
        for done_idx in np.where(dones)[0]:
            gradient_steps = self.gradient_steps or self.episode_steps[done_idx]
            self.train(int(gradient_steps))
        self.episode_steps = (self.episode_steps + 1) * (1 - dones)


if __name__ == '__main__':
    from utils import (IAmTheOtherKindOfReplayBufferBecauseFuckTensorflow,
                       ModelReader, create_gym_env)

    en = create_gym_env('BipedalWalker-v3', 1, False)
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
    agent = TD3(en, am, cm, bs, log_frequency=10)
    agent.fit(250)
