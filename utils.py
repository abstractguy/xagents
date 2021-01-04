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
    """
    Replay buffer that holds state transitions which supports:
    - N-step transitions.
    - Prioritized sampling.
    - Initial and max sizes.
    """

    def __init__(
        self,
        max_size,
        initial_size=None,
        n_steps=1,
        gamma=0.99,
        batch_size=32,
        prioritize=False,
        alpha=0.6,
        beta=0.4,
        beta_frames=100000,
        priority_bias=1e-5,
    ):
        """
        Initialize buffer settings.
        Args:
            max_size: Maximum transitions to store.
            initial_size: Maximum transitions to store before starting the training.
            n_steps: Steps separating start and end frames.
            gamma: Discount factor.
            batch_size: Size of the sampling method batch.
            prioritize: If True, Sampling will be prioritized.
            alpha: Alpha parameter to be used if prioritize==True.
            beta: Beta parameter to bes used if prioritize==True.
            beta_frames: Beta frames parameter to be used if prioritize==True.
            priority_bias: Bias to be added to the sample priorities if prioritize==True.
        """
        super(ReplayBuffer, self).__init__(maxlen=max_size)
        self.initial_size = initial_size or max_size
        self.n_steps = n_steps
        self.gamma = gamma
        self.temp_buffer = []
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.priorities = None
        if prioritize:
            self.priorities = deque(maxlen=max_size)
        self.priority_updates = 0
        self.beta_frames = beta_frames
        self.current_indices = None
        self.current_weights = None
        self.priority_bias = priority_bias

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
        if (self.temp_buffer and self.temp_buffer[-1][3]) or len(
            self.temp_buffer
        ) == self.n_steps:
            adjusted_sample = self.reset_temp_history()
            super(ReplayBuffer, self).append(adjusted_sample)
            if self.priorities is not None:
                priority = max(self.priorities, default=1)
                self.priorities.append(priority)
        self.temp_buffer.append(experience)

    def get_sample(self):
        """
        Get a sample of the replay buffer.
        Returns:
            A batch of observations in the form of
            [[states], [actions], [rewards], [dones], [next states]],
        """
        probabilities, weights = None, None
        if self.priorities is not None:
            probabilities = np.array(self.priorities) ** self.alpha
            probabilities /= probabilities.sum()
        self.current_indices = np.random.choice(
            len(self), self.batch_size, replace=False, p=probabilities
        )
        if isinstance(probabilities, np.ndarray):
            self.current_weights = (
                len(self) * probabilities[self.current_indices]
            ) ** (-self.beta)
            self.current_weights /= self.current_weights.max()
        memories = [self[i] for i in self.current_indices]
        batch = [np.array(item) for item in zip(*memories)]
        return batch

    def update_priorities(self, priorities):
        """
        Update sampling priorities and self.beta
        Args:
            priorities: numpy array of priorities post gradient update.

        Returns:
            None
        """
        for idx, priority in zip(self.current_indices, priorities):
            self.priorities[idx] = priority
        self.priority_updates += 1
        v = self.beta + self.priority_updates * (1.0 - self.beta) / self.beta_frames
        self.beta = min(1.0, v)


def create_gym_env(env_name, preprocess=True, *args, **kwargs):
    env = gym.make(env_name)
    if preprocess:
        env = AtariPreprocessor(env, *args, **kwargs)
    return env
