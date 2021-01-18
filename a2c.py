from collections import deque
from time import perf_counter

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import RMSprop

from models import CNNA2C
from utils import create_gym_env


class A2C:
    def __init__(self, envs, transition_steps=5, reward_buffer_size=100, gamma=0.99):
        assert envs, 'No environments given'
        self.envs = envs
        self.n_envs = len(envs)
        self.input_shape = self.envs[0].observation_space.shape
        self.n_actions = self.envs[0].action_space.n
        self.model = CNNA2C(self.input_shape, self.n_actions)
        self.transition_steps = transition_steps
        self.total_rewards = deque(maxlen=reward_buffer_size * self.n_envs)
        self.best_reward = -float('inf')
        self.mean_reward = -float('inf')
        self.states = np.zeros((self.n_envs, *self.input_shape))
        self.masks = np.ones(self.n_envs)
        self.reset_envs()
        self.steps = 0
        self.games = 0
        self.gamma = gamma
        self.episode_rewards = np.zeros(self.n_envs)
        self.last_reset_step = 0
        self.frame_speed = 0

    def reset_envs(self):
        for i, env in enumerate(self.envs):
            self.states[i] = env.reset()

    def update_returns(self, new_values, returns, masks, rewards):
        returns[-1] = new_values
        for step in reversed(range(self.transition_steps)):
            returns[step] = (
                returns[step + 1] * self.gamma * masks[step + 1] + rewards[step]
            )

    def step_envs(self, actions, done_envs):
        observations = []
        for (i, env), action in zip(enumerate(self.envs), actions):
            state, reward, done, _ = env.step(action)
            self.steps += 1
            observations.append((state, reward, done))
            self.episode_rewards[i] += reward
            if done:
                self.states[i] = env.reset()
                self.games += 1
                self.total_rewards.append(self.episode_rewards[i])
                self.episode_rewards[i] *= 0
                done_envs.append(1)
        return [np.array(item) for item in zip(*observations)]

    def play_steps(self, done_envs):
        state_b, action_b, log_prob_b, value_b, reward_b, mask_b = [
            [] for _ in range(6)
        ]
        state_b.append(self.states)
        mask_b.append(self.masks)
        for step in range(self.transition_steps):
            actions, log_probs, entropies, values = self.model(self.states)
            states, rewards, dones = self.step_envs(actions, done_envs)
            self.states = states
            self.masks[np.where(dones)] = 0
            state_b.append(states)
            action_b.append(actions)
            log_prob_b.append(log_probs)
            value_b.append(values)
            reward_b.append(rewards)
            mask_b.append(self.masks)
        *_, new_values = self.model(state_b[-1])
        results = new_values, state_b, action_b, log_prob_b, value_b, reward_b, mask_b
        return [np.array(item) for item in results]

    @tf.function
    def train_step(self, states, actions, returns):
        with tf.GradientTape() as tape:
            actions, log_probs, entropies, values = self.model(
                states, actions=actions, training=True
            )
            advantages = returns - values
            actor_loss = -tf.reduce_mean(advantages * log_probs)
            critic_loss = Huber()(values, returns)
            loss = actor_loss + critic_loss
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def display_metrics(self):
        display_titles = (
            'frame',
            'games',
            'speed',
            'mean reward',
            'best reward',
            'episode rewards',
        )
        display_values = (
            self.steps,
            self.games,
            f'{round(self.frame_speed)} steps/s',
            self.mean_reward,
            self.best_reward,
            list(self.total_rewards)[-self.n_envs :],
        )
        display = (
            f'{title}: {value}' for title, value in zip(display_titles, display_values)
        )
        print(', '.join(display))

    def update_metrics(self, start_time):
        """
        Update progress metrics.
        Args:
            start_time: Episode start time, used for calculating fps.
        Returns:
            None
        """
        self.frame_speed = (self.steps - self.last_reset_step) / (
            perf_counter() - start_time
        )
        self.last_reset_step = self.steps
        self.mean_reward = np.around(np.mean(self.total_rewards), 2)
        self.best_reward = max(self.mean_reward, self.best_reward)

    def fit(self, target_reward, learning_rate=7e-4):
        self.model.compile(RMSprop(learning_rate))
        returns = np.zeros((self.transition_steps + 1, self.n_envs), np.float32)
        done_envs = []
        start_time = perf_counter()
        while True:
            if len(done_envs) == self.n_envs:
                self.update_metrics(start_time)
                start_time = perf_counter()
                self.display_metrics()
                done_envs.clear()
            if self.mean_reward >= target_reward:
                print(f'Reward achieved in {self.steps} steps!')
                break
            new_values, *buffers = self.play_steps(done_envs)
            state_b, action_b, log_prob_b, value_b, reward_b, mask_b = buffers
            self.update_returns(new_values, returns, mask_b, reward_b)
            self.train_step(
                state_b[:-1].reshape(-1, *self.input_shape),
                action_b.reshape(-1),
                returns[:-1].reshape(-1),
            )


if __name__ == '__main__':
    en = create_gym_env('PongNoFrameskip-v4', 16)
    agn = A2C(en)
    agn.fit(18)
