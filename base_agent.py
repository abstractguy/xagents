from collections import deque
from datetime import timedelta
from time import perf_counter

import numpy as np


class BaseAgent:
    def __init__(
        self,
        envs,
        model,
        checkpoint=None,
        reward_buffer_size=100,
        transition_steps=1,
        gamma=0.99,
        metric_digits=2,
    ):
        """
        Base class for various types of agents.
        Args:
            envs: A list of gym environments that return states as atari frames.
            model: tf.keras.models.Model used for training.
            checkpoint: Path to .tf filename under which the trained model will be saved.
            reward_buffer_size: Size of the reward buffer that will hold the last n total
                rewards which will be used for calculating the mean reward.
            transition_steps: n-step transition for example given s1, s2, s3, s4 and n_step = 4,
                transition will be s1 -> s4 (defaults to 1, s1 -> s2)
            gamma: Discount factor used for gradient updates.
            metric_digits: Rounding decimals for display purposes.
        """
        assert envs, 'No Environments given'
        self.n_envs = len(envs)
        self.envs = envs
        self.checkpoint_path = checkpoint
        self.total_rewards = deque(maxlen=reward_buffer_size)
        self.transition_steps = transition_steps
        self.gamma = gamma
        self.model = model
        self.target_reward = None
        self.max_steps = None
        self.metric_digits = metric_digits
        self.input_shape = self.envs[0].observation_space.shape
        self.available_actions = self.envs[0].action_space.n
        self.best_reward = -float('inf')
        self.mean_reward = -float('inf')
        self.states = [None] * self.n_envs
        self.steps = 0
        self.frame_speed = 0
        self.last_reset_step = 0
        self.training_start_time = None
        self.last_reset_time = None
        self.games = 0
        self.episode_rewards = np.zeros(self.n_envs)
        self.episode_scores = deque(maxlen=self.n_envs)
        self.done_envs = []
        self.reset_envs()

    def reset_envs(self):
        """
        Reset all environments in self.envs
        Returns:
            None
        """
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
        """
        Display progress metrics to the console.
        Returns:
            None
        """
        display_titles = (
            'steps',
            'games',
            'speed',
            'mean reward',
            'time',
            'best reward',
            'episode rewards',
        )
        display_values = (
            self.steps,
            self.games,
            f'{round(self.frame_speed)} steps/s',
            self.mean_reward,
            timedelta(seconds=perf_counter() - self.training_start_time),
            self.best_reward,
            [*self.episode_scores],
        )
        display = (
            f'{title}: {value}' for title, value in zip(display_titles, display_values)
        )
        print(', '.join(display))

    def update_metrics(self):
        """
        Update progress metrics.
        Returns:
            None
        """
        self.checkpoint()
        self.frame_speed = (self.steps - self.last_reset_step) / (
            perf_counter() - self.last_reset_time
        )
        self.last_reset_step = self.steps
        self.mean_reward = np.around(np.mean(self.total_rewards), self.metric_digits)

    def check_episodes(self):
        """
        Check environment done counts to display progress and update metrics.
        Returns:
            None
        """
        if len(self.done_envs) == self.n_envs:
            self.update_metrics()
            self.last_reset_time = perf_counter()
            self.display_metrics()
            self.done_envs.clear()

    def training_done(self):
        """
        Check if the training is done by a target reward or maximum number of steps.
        Returns:
            bool
        """
        if self.mean_reward >= self.target_reward:
            print(f'Reward achieved in {self.steps} steps!')
            return True
        if self.max_steps and self.steps >= self.max_steps:
            print(f'Maximum steps exceeded')
            return True
        return False

    def step_envs(self, actions):
        """
        Play 1 step for each env in self.envs
        Args:
            actions: numpy array / list of actions.
        Returns:
            numpy array of [[states], [rewards], [dones]] or a numpy placeholder
            for compatibility with tf.numpy_function()
        """
        observations = []
        for (i, env), action in zip(enumerate(self.envs), actions):
            state = self.states[i]
            new_state, reward, done, _ = env.step(action)
            self.states[i] = new_state
            self.steps += 1
            self.episode_rewards[i] += reward
            if hasattr(self, 'buffers'):
                self.buffers[i].append((state, action, reward, done, new_state))
            else:
                observations.append((new_state, reward, done))
            if done:
                self.done_envs.append(1)
                self.total_rewards.append(self.episode_rewards[i])
                self.episode_scores.append(self.episode_rewards[i])
                self.games += 1
                self.episode_rewards[i] = 0
                self.states[i] = env.reset()
        if hasattr(self, 'buffers'):
            return [
                np.array([1], np.float32),
                np.array([1], np.float32),
                np.array([1], np.bool),
            ]
        return [np.array(item, np.float32) for item in zip(*observations)]
