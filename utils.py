import random
from collections import deque

import cv2
import gym
import numpy as np


class AtariPreprocessor(gym.Wrapper):
    """
    gym wrapper for preprocessing atari frames.
    """

    def __init__(self, env, frame_skips=4, resize_shape=(84, 84), state_buffer_size=2):
        """
        Initialize preprocessing settings.
        Args:
            env: gym environment that returns states as atari frames.
            frame_skips: Number of frame skips to use per environment step.
            resize_shape: (m, n) output frame size.
        """
        assert frame_skips > 1, 'frame_skips must be >= 1'
        super(AtariPreprocessor, self).__init__(env)
        self.skips = frame_skips
        self.frame_shape = resize_shape
        self.observation_space.shape = (*resize_shape, 1)
        self.observation_buffer = deque(maxlen=state_buffer_size)

    def process_frame(self, frame):
        """
        Resize and convert atari frame to grayscale.
        Args:
            frame: Image as numpy.ndarray

        Returns:
            Processed frame.
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, self.frame_shape) / 255
        return np.expand_dims(frame, -1)

    def step(self, action: int):
        """
        Step respective to self.skips.
        Args:
            action: Action supported by self.env

        Returns:
            (state, reward, done, info)
        """
        total_reward = 0
        state, done, info = 3 * [None]
        max_frame = None
        for _ in range(self.skips):
            state, reward, done, info = self.env.step(action)
            self.observation_buffer.append(state)
            max_frame = np.max(np.stack(self.observation_buffer), axis=0)
            total_reward += reward
            if done:
                break
        return self.process_frame(max_frame), total_reward, done, info

    def reset(self, **kwargs):
        """
        Reset self.env
        Args:
            **kwargs: kwargs passed to self.env.reset()

        Returns:
            Processed atari frame.
        """
        self.observation_buffer.clear()
        observation = self.env.reset(**kwargs)
        self.observation_buffer.append(observation)
        return self.process_frame(observation)


class ReplayBuffer(deque):
    """
    Replay buffer that holds state transitions
    """

    def __init__(
        self,
        max_size,
        initial_size=None,
        n_steps=1,
        gamma=0.99,
        batch_size=32,
        seed=None,
    ):
        """
        Initialize buffer settings.
        Args:
            max_size: Maximum transitions to store.
            initial_size: Maximum transitions to store before starting the training.
            n_steps: Steps separating start and end frames.
            gamma: Discount factor.
            batch_size: Size of the sampling method batch.
        """
        super(ReplayBuffer, self).__init__(maxlen=max_size)
        self.initial_size = initial_size or max_size
        self.n_steps = n_steps
        self.gamma = gamma
        self.temp_buffer = []
        self.batch_size = batch_size
        if seed:
            random.seed(seed)

    def reset_temp_history(self):
        """
        Calculate start and end frames and clear temp buffer.

        Returns:
            state, action, reward, done, new_state
        """
        reward = 0
        for exp in self.temp_buffer[::-1]:
            reward *= self.gamma
            reward += exp[2]
        state = self.temp_buffer[0][0]
        action = self.temp_buffer[0][1]
        done = self.temp_buffer[-1][3]
        new_state = self.temp_buffer[-1][-1]
        self.temp_buffer.clear()
        return state, action, reward, done, new_state

    def append(self, experience):
        """
        Append experience and auto-allocate to temp buffer / main buffer(self)
        Args:
            experience: state, action, reward, done, new_state

        Returns:
            None
        """
        if self.n_steps == 1:
            super(ReplayBuffer, self).append(experience)
            return
        if (self.temp_buffer and self.temp_buffer[-1][3]) or len(
            self.temp_buffer
        ) == self.n_steps:
            adjusted_sample = self.reset_temp_history()
            super(ReplayBuffer, self).append(adjusted_sample)
        self.temp_buffer.append(experience)

    def get_sample(self):
        """
        Get a sample of the replay buffer.

        Returns:
            A batch of observations in the form of
            [[states], [actions], [rewards], [dones], [next states]],
        """
        memories = random.sample(self, self.batch_size)
        if self.batch_size > 1:
            return [np.array(item) for item in zip(*memories)]
        return memories[0]


def create_gym_env(env_name, n=1, preprocess=True, *args, **kwargs):
    """
    Create gym environment and initialize preprocessing settings.
    Args:
        env_name: Name of the environment to be passed to gym.make()
        n: Number of environments to create.
        preprocess: If True, AtariPreprocessor will be used.
        *args: args to be passed to AtariPreprocessor
        **kwargs: kwargs to be passed to AtariPreprocessor

    Returns:
        A list of gym environments.
    """
    envs = [gym.make(env_name) for _ in range(n)]
    if preprocess:
        envs = [AtariPreprocessor(env, *args, **kwargs) for env in envs]
    return envs
