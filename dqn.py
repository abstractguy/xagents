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
    def __init__(self, env, buffer_size=10000, batch_size=32):
        self.env = wrappers.make_env(env)
        self.input_shape = self.env.observation_space.shape
        self.main_model = self.create_model()
        self.target_model = self.create_model()
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def create_model(self):
        x0 = Input(self.input_shape)
        x = Conv2D(32, 8, 4, activation='relu')(x0)
        x = Conv2D(64, 4, 2, activation='relu')(x)
        x = Conv2D(64, 3, 1, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(512, 'relu')(x)
        x = Dense(self.env.action_space.n)(x)
        return Model(x0, x)

    def get_action(self, epsilon, state):
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        q_values = self.main_model.predict(np.expand_dims(state, 0))
        return np.argmax(q_values)

    def get_buffer_sample(self):
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        memories = [self.buffer[i] for i in indices]
        batch = [np.array(item) for item in zip(*memories)]
        return batch

    @tf.function
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
        # q_new_states = self.main_model.predict(new_states)
        # q_target = np.copy(q_states)
        # idx = np.arange(self.batch_size, dtype=np.uint8)
        # q_target[idx, actions] = rewards + gamma * np.max(q_new_states, axis=1) * dones
        # self.main_model.fit(states, q_target, verbose=0)

    def fit(
        self,
        epsilon_start=1,
        epsilon_end=0.01,
        decay_n_frames=150000,
        lr=1e-4,
        gamma=0.99,
        target_reward=19,
        update_target_frames=1000,
    ):
        state = self.env.reset()
        total_rewards = []
        episode_reward = 0
        frame_idx = 0
        start_time = perf_counter()
        optimizer = Adam(lr)
        self.main_model.compile(optimizer, loss='mse')
        mean_reward = 0
        last_reset_frame = 0
        display_titles = ('frame', 'games', 'reward', 'epsilon', 'speed')
        while True:
            frame_idx += 1
            epsilon = max(epsilon_end, epsilon_start - frame_idx / decay_n_frames)
            action = self.get_action(epsilon, state)
            new_state, reward, done, info = self.env.step(action)
            episode_reward += reward
            self.buffer.append((state, action, reward, done, new_state))
            state = new_state
            if done:
                if mean_reward >= target_reward:
                    print(f'Reward achieved in {frame_idx} frames!')
                    break
                total_rewards.append(episode_reward)
                speed = (frame_idx - last_reset_frame) / (perf_counter() - start_time)
                last_reset_frame = frame_idx
                start_time = perf_counter()
                mean_reward = np.mean(total_rewards[-100:])
                display_values = (
                    frame_idx,
                    len(total_rewards),
                    np.around(mean_reward, 2),
                    np.around(epsilon, 2),
                    f'{round(speed)} frames/s',
                )
                display = (
                    f'{title}: {value}'
                    for title, value in zip(display_titles, display_values)
                )
                print(*display, sep=', ')
                episode_reward = 0
                state = self.env.reset()
            if len(self.buffer) >= self.buffer_size:
                batch = self.get_buffer_sample()
                self.update(batch, gamma)
            if frame_idx % update_target_frames == 0:
                self.target_model.set_weights(self.main_model.get_weights())


if __name__ == '__main__':
    # tf.compat.v1.disable_eager_execution()
    # wandb.init(name='Post modification test')
    activate_gpu_tf()
    agn = DQN('PongNoFrameskip-v4', 100)
    agn.fit()
