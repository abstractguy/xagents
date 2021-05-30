from collections import deque
from datetime import timedelta
from time import perf_counter

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input


def actor(input_shape, output_units):
    x0 = Input(input_shape)
    x = Dense(400, 'relu')(x0)
    x = Dense(300, 'relu')(x)
    output = Dense(output_units, 'tanh')(x)
    model = Model(x0, output)
    model.call = tf.function(model.call)
    return model


def critic(input_shape):
    x0 = Input(input_shape)
    x = Dense(400, 'relu')(x0)
    x = Dense(300, 'relu')(x)
    ouput = Dense(1)(x)
    model = Model(x0, ouput)
    model.call = tf.function(model.call)
    return model


class BaseBuffer:
    def __init__(self, buffer_size, obs_dim, action_dim, n_envs=1):
        super(BaseBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.pos = 0
        self.full = False
        self.n_envs = n_envs

    def size(self):
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs):
        raise NotImplementedError()

    def reset(self):
        self.pos = 0
        self.full = False

    def sample(self, batch_size):
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds):
        raise NotImplementedError()


class ReplayBuffer(BaseBuffer):
    def __init__(self, buffer_size, obs_dim, action_dim, n_envs=1):
        super(ReplayBuffer, self).__init__(
            buffer_size, obs_dim, action_dim, n_envs=n_envs
        )
        assert n_envs == 1
        self.observations = np.zeros(
            (self.buffer_size, self.n_envs, self.obs_dim), dtype=np.float32
        )
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32
        )
        self.next_observations = np.zeros(
            (self.buffer_size, self.n_envs, self.obs_dim), dtype=np.float32
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done):
        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _get_samples(self, batch_inds):
        return (
            self.observations[batch_inds, 0, :],
            self.actions[batch_inds, 0, :],
            self.next_observations[batch_inds, 0, :],
            self.dones[batch_inds],
            self.rewards[batch_inds],
        )


class TD3:
    def __init__(
        self,
        env,
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
        self.actor = actor(env.observation_space.shape, env.action_space.shape[0])
        self.critic1 = critic(
            env.observation_space.shape[0] + env.action_space.shape[0]
        )
        self.critic2 = tf.keras.models.clone_model(self.critic1)
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
        for model in (self.actor, self.critic1, self.critic2):
            model.compile('adam')
        self.replay_buffer = ReplayBuffer(
            self.buffer_size, env.observation_space.shape[0], env.action_space.shape[0]
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

    def train(self, gradient_steps, batch_size=100, policy_delay=2):
        for gradient_step in range(gradient_steps):
            obs, action, next_obs, done, reward = self.replay_buffer.sample(batch_size)
            self._train_critic(obs, action, next_obs, done, reward)
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
                replay_buffer=self.replay_buffer,
                obs=obs,
                log_interval=log_interval,
            )
            if self.steps > 0 and self.steps > self.learning_starts:
                self.train(
                    self.gradient_steps or self.last_episode_steps[0],
                    batch_size=self.batch_size,
                    policy_delay=self.policy_delay,
                )
        return self

    def collect_rollouts(
        self,
        env,
        learning_starts=0,
        num_timesteps=0,
        replay_buffer=None,
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
            if replay_buffer is not None:
                replay_buffer.add(obs, new_obs, action, reward, done_bool)
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
    env = gym.make('BipedalWalker-v3')
    agent = TD3(env)
    agent.learn(1000000, log_interval=10)
