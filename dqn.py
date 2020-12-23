from collections import deque
from time import perf_counter

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import wandb
from Chapter06.lib import wrappers
from utils import activate_gpu_tf


class DQN:
    def __init__(
        self,
        env,
        buffer_size=10000,
        batch_size=32,
        checkpoint=None,
        target_reward=19,
        reward_mean_n=100,
        epsilon_start=1,
        epsilon_end=0.01,
    ):
        self.env = wrappers.make_env(env)
        self.input_shape = self.env.observation_space.shape
        self.main_model = self.create_model()
        self.target_model = self.create_model()
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint
        self.total_rewards = []
        self.best_reward = 0
        self.mean_reward = 0
        self.target_reward = target_reward
        self.reward_mean_n = reward_mean_n
        self.state = self.env.reset()
        self.frame_idx = 0
        self.frame_speed = 0
        self.last_reset_frame = 0
        self.epsilon_start = self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end

    def create_model(self):
        x0 = Input(self.input_shape)
        x = Conv2D(32, 8, 4, activation='relu')(x0)
        x = Conv2D(64, 4, 2, activation='relu')(x)
        x = Conv2D(64, 3, 1, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(512, 'relu')(x)
        x = Dense(self.env.action_space.n)(x)
        return Model(x0, x)

    def get_action(self):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        q_values = self.main_model.predict(np.expand_dims(self.state, 0))
        return np.argmax(q_values)

    def get_buffer_sample(self):
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        memories = [self.buffer[i] for i in indices]
        batch = [np.array(item) for item in zip(*memories)]
        return batch

    def update(self, batch, gamma):
        states, actions, rewards, dones, new_states = batch
        q_states = self.main_model.predict(states)
        new_state_values = self.target_model.predict(new_states).max(1)
        new_state_values[dones] = 0
        target_values = np.copy(q_states)
        target_values[np.arange(self.batch_size), actions] = (
            new_state_values * gamma + rewards
        )
        self.main_model.fit(states, target_values, verbose=0)

    def checkpoint(self):
        if self.best_reward < self.mean_reward:
            print(f'Best reward updated: {self.best_reward} -> {self.mean_reward}')
            if self.checkpoint_path:
                self.main_model.save(self.checkpoint_path)
            self.best_reward = self.best_reward

    def display_metrics(self):
        display_titles = ('frame', 'games', 'reward', 'epsilon', 'speed')
        display_values = (
            self.frame_idx,
            len(self.total_rewards),
            np.around(self.mean_reward, 2),
            np.around(self.epsilon, 2),
            f'{round(self.frame_speed)} frames/s',
        )
        display = (
            f'{title}: {value}' for title, value in zip(display_titles, display_values)
        )
        print(*display, sep=', ')

    def reset(self, episode_reward, start_time):
        self.total_rewards.append(episode_reward)
        self.frame_speed = (self.frame_idx - self.last_reset_frame) / (
            perf_counter() - start_time
        )
        self.last_reset_frame = self.frame_idx
        self.mean_reward = np.mean(self.total_rewards[-self.reward_mean_n :])

    def fit(
        self,
        decay_n_frames=150000,
        lr=1e-4,
        gamma=0.99,
        update_target_frames=1000,
    ):
        episode_reward = 0
        start_time = perf_counter()
        optimizer = Adam(lr)
        self.main_model.compile(optimizer, loss='mse')
        self.target_model.compile(optimizer, loss='mse')
        while True:
            self.frame_idx += 1
            self.epsilon = max(
                self.epsilon_end, self.epsilon_start - self.frame_idx / decay_n_frames
            )
            action = self.get_action()
            new_state, reward, done, info = self.env.step(action)
            episode_reward += reward
            self.buffer.append((self.state, action, reward, done, new_state))
            self.state = new_state
            if done:
                if self.mean_reward >= self.target_reward:
                    print(f'Reward achieved in {self.frame_idx} frames!')
                    break
                self.checkpoint()
                self.reset(episode_reward, start_time)
                self.display_metrics()
                start_time = perf_counter()
                episode_reward = 0
                self.state = self.env.reset()
            if len(self.buffer) < self.buffer_size:
                continue
            batch = self.get_buffer_sample()
            self.update(batch, gamma)
            if self.frame_idx % update_target_frames == 0:
                self.target_model.set_weights(self.main_model.get_weights())


if __name__ == '__main__':
    # tf.compat.v1.disable_eager_execution()
    # wandb.init(name='Post modification test')
    activate_gpu_tf()
    agn = DQN('PongNoFrameskip-v4', 10000)
    agn.fit()
