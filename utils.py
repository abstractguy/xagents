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
            state_buffer_size: State buffer for max pooling.
        """
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
        for _ in range(self.skips):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            self.observation_buffer.append(state)
            if done:
                break
        max_frame = np.max(np.stack(self.observation_buffer), axis=0)
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
    def __init__(self, size, n_steps=1, gamma=0.99, batch_size=32):
        super(ReplayBuffer, self).__init__(maxlen=size)
        self.n_steps = n_steps
        self.gamma = gamma
        self.temp_history = []
        self.batch_size = batch_size

    def append(self, experience):
        total_reward = 0
        if (self.temp_history and self.temp_history[-1][3]) or len(
            self.temp_history
        ) == self.n_steps:
            for exp in self.temp_history[::-1]:
                total_reward *= self.gamma
                total_reward += exp[2]
            state = self.temp_history[0][0]
            action = self.temp_history[0][1]
            done = self.temp_history[-1][3]
            new_state = self.temp_history[-1][-1]
            super(ReplayBuffer, self).append(
                (state, action, total_reward, done, new_state)
            )
            self.temp_history.clear()
        self.temp_history.append(experience)

    def get_sample(self):
        """
        Get a sample of the replay buffer.
        Returns:
            A batch of observations in the form of
            [[states], [actions], [rewards], [dones], [next states]]
        """
        indices = np.random.choice(len(self), self.batch_size, replace=False)
        memories = [self[i] for i in indices]
        batch = [np.array(item) for item in zip(*memories)]
        return batch
