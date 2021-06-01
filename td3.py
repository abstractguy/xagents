from collections import deque
from datetime import timedelta
from time import perf_counter

import gym
import numpy as np
import tensorflow as tf

from utils import IAmTheOtherKindOfReplayBufferBecauseFuckTensorflow


class TD3:
    def __init__(
        self,
        env,
        actor_model,
        critic_model,
        buffer_size=int(1e6),
        learning_rate=1e-3,
        policy_delay=2,
        learning_starts=100,
        gamma=0.99,
        batch_size=100,
        train_freq=-1,
        gradient_steps=None,
        n_episodes_rollout=1,
        tau=0.005,
        action_noise=None,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        _init_setup_model=True,
    ):
        self.num_timesteps = 0
        self.env = env
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.n_episodes_rollout = n_episodes_rollout
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.action_noise = action_noise
        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise
        self.total_rewards = deque(maxlen=100)
        self.best_reward = -float('inf')
        self.mean_reward = -float('inf')
        self.last_reset_step = 0
        self.last_reset_time = None
        self.training_start_time = None
        self.episode_reward = 0
        self.episode_steps = 0
        self.last_episode_steps = deque(maxlen=1)
        self.steps = 0
        self.games = 0
        self.display_titles = (
            'time',
            'steps',
            'games',
            'fps',
            'mean reward',
            'best reward',
        )
        self.actor = actor_model
        self.critic1 = critic_model
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
        self.replay_buffer = IAmTheOtherKindOfReplayBufferBecauseFuckTensorflow(
            buffer_size, 5, batch_size=batch_size, initial_size=learning_starts
        )

    def critic_loss(self, obs, action, next_obs, done, reward):
        noise = tf.random.normal(action.shape) * self.target_policy_noise
        noise = tf.clip_by_value(noise, -self.target_noise_clip, self.target_noise_clip)
        next_action = tf.clip_by_value(self.target_actor(next_obs) + noise, -1.0, 1.0)
        target_critic_input = tf.concat([next_obs, next_action], 1)
        target_q1 = self.target_critic1(target_critic_input)
        target_q2 = self.target_critic2(target_critic_input)
        target_q = tf.minimum(target_q1, target_q2)
        target_q = reward + tf.stop_gradient((1 - done) * self.gamma * target_q)
        critic_input = tf.concat([obs, action], 1)
        current_q1 = self.critic1(critic_input)
        current_q2 = self.critic2(critic_input)
        return tf.keras.losses.MSE(current_q1, target_q), tf.keras.losses.MSE(
            current_q2, target_q
        )

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

    def train(self, gradient_steps, policy_delay=2):
        for gradient_step in range(gradient_steps):
            obs, new_obs, action, reward, done = self.replay_buffer.get_sample()
            self._train_critic(obs, action, new_obs, done, reward)
            if gradient_step % policy_delay == 0:
                self._train_actor(obs)
                self.update_targets()

    def learn(
        self,
        total_timesteps,
        log_interval=4,
    ):
        obs = self.env.reset()
        self.training_start_time = perf_counter()
        self.last_reset_time = self.training_start_time
        while self.num_timesteps < total_timesteps:
            self.collect_rollouts(
                self.env,
                learning_starts=self.learning_starts,
                num_timesteps=self.num_timesteps,
                obs=obs,
                log_interval=log_interval,
            )
            if self.steps > 0 and self.steps > self.learning_starts:
                self.train(
                    self.gradient_steps or self.last_episode_steps[0],
                    policy_delay=self.policy_delay,
                )
        return self

    def collect_rollouts(
        self,
        env,
        learning_starts=0,
        num_timesteps=0,
        obs=None,
        log_interval=1,
    ):
        while True:
            if self.steps < learning_starts:
                action = env.action_space.sample()
            else:
                action = np.squeeze(self.actor(np.expand_dims(obs, 0)))
            new_obs, reward, done, infos = env.step(action)
            done_bool = float(done)
            self.episode_reward += reward
            self.replay_buffer.append(obs, new_obs, action, reward, done_bool)
            obs = new_obs
            num_timesteps += 1
            self.steps += 1
            self.episode_steps += 1
            if done:
                env.reset()
                current_time = perf_counter()
                self.games += 1
                interval_steps = self.steps - self.last_reset_step
                interval_time = current_time - self.last_reset_time
                fps = interval_steps // interval_time
                self.total_rewards.append(self.episode_reward)
                self.mean_reward = int(np.mean(self.total_rewards))
                self.best_reward = int(max(self.episode_reward, self.best_reward))
                time_so_far = timedelta(seconds=current_time - self.training_start_time)
                self.episode_reward = 0
                self.last_episode_steps.append(self.episode_steps)
                self.episode_steps = 0
                self.last_reset_time = current_time
                self.last_reset_step = self.steps
                if self.games % log_interval == 0:
                    self.display_titles = (
                        'time',
                        'steps',
                        'games',
                        'fps',
                        'mean reward',
                        'best reward',
                    )
                    values = (
                        time_so_far,
                        self.steps,
                        self.games,
                        fps,
                        self.mean_reward,
                        self.best_reward,
                    )
                    print(
                        ', '.join(
                            f'{title}: {value}'
                            for (title, value) in zip(self.display_titles, values)
                        )
                    )
                break


if __name__ == '__main__':
    from utils import ModelReader

    en = gym.make('BipedalWalker-v3')
    amr = ModelReader(
        'models/ann/td3-actor.cfg',
        [en.action_space.shape[0]],
        en.observation_space.shape,
        'adam',
    )
    cmr = ModelReader(
        'models/ann/td3-critic.cfg',
        [1],
        en.observation_space.shape[0] + en.action_space.shape[0],
        'adam',
    )
    am = amr.build_model()
    cm = cmr.build_model()
    agent = TD3(en, am, cm)
    agent.learn(1000000, log_interval=10)
