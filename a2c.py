from collections import deque

import numpy as np

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

    def play_steps(self):
        state_b, action_b, log_prob_b, value_b, reward_b, mask_b = [
            [] for _ in range(6)
        ]
        state_b.append(self.states)
        mask_b.append(self.masks)
        done_envs = []
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
        return new_values, state_b, action_b, log_prob_b, value_b, reward_b, mask_b

    def fit(self):
        returns = np.zeros((self.n_actions, self.n_envs))
        new_values, *buffers = self.play_steps()
        state_b, action_b, log_prob_b, value_b, reward_b, mask_b = buffers
        self.update_returns(new_values, returns, mask_b, reward_b)
        actions, log_probs, entropies, values = self.model(
            np.array(state_b[:-1]).reshape(-1, *self.input_shape),
            actions=np.array(action_b).reshape(-1),
        )
        print(actions)
        print(log_probs)
        print(entropies)
        print(values)
        pass


if __name__ == '__main__':
    en = create_gym_env('PongNoFrameskip-v4', 16)
    agn = A2C(en)
    agn.fit()
