from collections import deque
from time import perf_counter

import numpy as np

from utils import create_gym_env


class BaseAgent:
    def __init__(
        self,
        envs,
        model,
        target_reward,
        checkpoint=None,
        reward_buffer_size=100,
        transition_steps=1,
        gamma=0.99,
        max_steps=None,
        metrics='all',
        metric_digits=2,
    ):
        assert envs, 'No Environments given'
        self.n_envs = len(envs)
        self.envs = envs
        self.checkpoint_path = checkpoint
        self.total_rewards = deque(maxlen=reward_buffer_size)
        self.transition_steps = transition_steps
        self.gamma = gamma
        self.model = model
        self.target_reward = target_reward
        self.max_steps = max_steps
        self.metrics = metrics
        self.metric_digits = metric_digits
        self.input_shape = self.envs[0].observation_space.shape
        self.available_actions = self.envs[0].action_space.n
        self.best_reward = [-float('inf')]
        self.mean_reward = [-float('inf')]
        self.states = [None] * self.n_envs
        self.steps = [0]
        self.frame_speed = [0]
        self.last_reset_step = 0
        self.training_start_time = None
        self.last_reset_time = None
        self.games = [0]
        self.episode_rewards = np.zeros(self.n_envs)
        self.episode_last_scores = deque(maxlen=self.n_envs)
        self.done_envs = []
        self.display_options = {
            'frames': self.steps,
            'games': self.games,
            'fps': self.frame_speed,
            'mean_reward': self.mean_reward,
            'best_reward': self.best_reward,
            'episode_rewards': self.episode_last_scores,
        }
        assert [
            item in self.display_options for item in metrics
        ], f'One or more given metrics is invalid'
        self.reset_envs()

    def reset_envs(self):
        for i, env in enumerate(self.envs):
            self.states[i] = env.reset()

    def checkpoint(self):
        """
        Save model weights if current reward > best reward.
        Returns:
            None
        """
        if self.best_reward < self.mean_reward:
            print(f'Best reward updated: {self.best_reward} -> {self.mean_reward}')
            if self.checkpoint_path:
                self.model.save_weights(self.checkpoint_path)
        self.best_reward = max(self.mean_reward, self.best_reward)

    def display_metrics(self):
        display = (
            f'{title.replace("_", " ")}: {round(value[0], self.metric_digits)}'
            for title, value in self.display_options.items()
            if title in self.metrics or self.metrics == 'all'
        )
        print(', '.join(display))

    def update_metrics(self):
        """
        Update progress metrics.
        Returns:
            None
        """
        self.checkpoint()
        self.frame_speed = (self.steps[0] - self.last_reset_step) / (
            perf_counter() - self.last_reset_time
        )
        self.last_reset_step = self.steps[0]
        self.mean_reward = np.around(np.mean(self.total_rewards), 2)

    def training_done(self):
        if self.mean_reward >= self.target_reward:
            print(f'Reward achieved in {self.steps} steps!')
            return True
        if self.max_steps and self.steps >= self.max_steps:
            print(f'Maximum steps exceeded')
            return True
        return False

    def check_episodes(self):
        if len(self.done_envs) == self.n_envs:
            self.update_metrics()
            self.last_reset_time = perf_counter()
            self.display_metrics()
            self.done_envs.clear()

    def step_envs(self, actions):
        """
        Play 1 step for each env in self.envs
        Args:
            actions: numpy array / list of actions.
        """
        buffer_samples = []
        observations = []
        for (i, env), action in zip(enumerate(self.envs), actions):
            state = self.states[i]
            new_state, reward, done, _ = env.step(action)
            self.steps += 1
            self.episode_rewards[i] += reward
            self.states[i] = new_state
            if hasattr(self, 'buffers'):
                buffer = self.buffers[i]
                buffer.append((state, action, reward, done, new_state))
                buffer_sample = buffer.get_sample()
                buffer_samples.append(buffer_sample)
            else:
                observations.append((state, reward, done))
            if done:
                self.done_envs.append(1)
                self.total_rewards.append(self.episode_rewards[i])
                self.games += 1
                self.episode_rewards[i] = 0
                self.states[i] = env.reset()
        if buffer_samples:
            if len(buffer_samples) == 1:
                return buffer_samples[0]
            return [np.concatenate(item) for item in zip(*buffer_samples)]
        return [np.array(item, np.float32) for item in zip(*observations)]
