import configparser
import random
from collections import deque

import cv2
import gym
import numpy as np
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model


class AtariPreprocessor(gym.Wrapper):
    """
    gym wrapper for preprocessing atari frames.
    """

    def __init__(
        self,
        env,
        frame_skips=4,
        resize_shape=(84, 84),
        state_buffer_size=2,
        scale_frames=True,
    ):
        """
        Initialize preprocessing settings.
        Args:
            env: gym environment that returns states as atari frames.
            frame_skips: Number of frame skips to use per environment step.
            resize_shape: (m, n) output frame size.
            state_buffer_size: Buffer size which is used to hold frames during steps.
            scale_frames: If False, frames will not be scaled / normalized (divided by 255)
        """
        assert frame_skips > 1, 'frame_skips must be >= 1'
        super(AtariPreprocessor, self).__init__(env)
        self.skips = frame_skips
        self.frame_shape = resize_shape
        self.observation_space.shape = (*resize_shape, 1)
        self.observation_buffer = deque(maxlen=state_buffer_size)
        self.scale_frames = scale_frames

    def process_frame(self, frame):
        """
        Resize and convert atari frame to grayscale.
        Args:
            frame: Atari frame as numpy.ndarray

        Returns:
            Processed frame.
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, self.frame_shape)
        if self.scale_frames:
            frame = frame / 255
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


class BaseBuffer:
    def __init__(self, size, initial_size=None, batch_size=32):
        if initial_size:
            assert size >= initial_size, 'Buffer initial size exceeds max size'
        self.size = size
        self.initial_size = initial_size or size
        self.batch_size = batch_size
        self.current_size = 0

    def append(self, *args):
        raise NotImplementedError(
            f'append() should be implemented by {self.__class__.__name__} subclasses'
        )

    def get_sample(self):
        raise NotImplementedError(
            f'get_sample() should be implemented by {self.__class__.__name__} subclasses'
        )


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer that holds state transitions
    """

    def __init__(self, size, n_steps=1, gamma=0.99, **kwargs):
        """
        Initialize buffer settings.
        Args:
            n_steps: Steps separating start and end frames.
            gamma: Discount factor.
        """
        super(ReplayBuffer, self).__init__(size, **kwargs)
        self.n_steps = n_steps
        self.gamma = gamma
        self.main_buffer = deque(maxlen=size)
        self.temp_buffer = []

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

    def append(self, *args):
        """
        Append experience and auto-allocate to temp buffer / main buffer(self)
        Args:
            *args: Items to store

        Returns:
            None
        """
        self.current_size += 1
        if self.n_steps == 1:
            self.main_buffer.append(args)
            return
        if (self.temp_buffer and self.temp_buffer[-1][3]) or len(
            self.temp_buffer
        ) == self.n_steps:
            adjusted_sample = self.reset_temp_history()
            self.main_buffer.append(adjusted_sample)
        self.temp_buffer.append(args)

    def get_sample(self):
        """
        Get a sample of the replay buffer.

        Returns:
            A batch of observations that has the same length as the
                previously appended experience.
        """
        memories = random.sample(self.main_buffer, self.batch_size)
        if self.batch_size > 1:
            return [np.array(item) for item in zip(*memories)]
        return memories[0]


class IAmTheOtherKindOfReplayBufferBecauseFuckTensorflow(BaseBuffer):
    def __init__(self, size, slots, **kwargs):
        super(IAmTheOtherKindOfReplayBufferBecauseFuckTensorflow, self).__init__(
            size, **kwargs
        )
        self.slots = [np.array([])] * slots
        self.current_size = 0

    def append(self, *args):
        self.current_size += 1
        for i, arg in enumerate(args):
            if not isinstance(arg, np.ndarray):
                arg = np.array([arg])
            if not self.slots[i].shape[0]:
                self.slots[i] = np.zeros((self.size, *arg.shape), np.float32)
            self.slots[i][self.current_size % self.size] = arg.copy()

    def get_sample(self):
        indices = np.random.randint(
            0, min(self.current_size, self.size), self.batch_size
        )
        return [slot[indices] for slot in self.slots]


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


class ModelReader:
    """
    Model utility class to create keras models from configuration files.
    """

    def __init__(self, cfg_file, output_units, input_shape, optimizer=None, seed=None):
        """
        Initialize model parser.
        Args:
            cfg_file: Path to .cfg file having that will be created.
            output_units: A list of output units that must be of the
                same size as the number of dense layers in the configuration
                without specified units.
            optimizer: tf.keras.optimizers.Optimizer
            seed: Random seed used by layer initializers.
        """
        self.initializers = {'orthogonal': Orthogonal}
        with open(cfg_file) as cfg:
            self.parser = configparser.ConfigParser()
            self.parser.read_file(cfg)
        self.optimizer = optimizer
        self.output_units = output_units
        self.input_shape = input_shape
        self.seed = seed
        self.output_count = 0

    def get_initializer(self, section):
        """
        Get layer initializer if specified in the configuration.
        Args:
            section: str, representing section unique name.

        Returns:
            tf.keras.initializers.Initializer
        """
        initializer_name = self.parser[section].get('initializer')
        gain = self.parser[section].get('gain')
        initializer_kwargs = {'seed': self.seed}
        if gain:
            initializer_kwargs.update({'gain': float(gain)})
        initializer = self.initializers.get(initializer_name)
        if initializer:
            return initializer(**initializer_kwargs)

    def create_convolution(self, section):
        """
        Parse convolution layer parameters and create layer.
        Args:
            section: str, representing section unique name.

        Returns:
            tf.keras.layers.Conv2D
        """
        filters = int(self.parser[section]['filters'])
        kernel_size = int(self.parser[section]['size'])
        stride = int(self.parser[section]['stride'])
        activation = self.parser[section].get('activation')
        return Conv2D(
            filters,
            kernel_size,
            stride,
            activation=activation,
            kernel_initializer=self.get_initializer(section),
        )

    def create_dense(self, section):
        """
        Parse dense layer parameters and create layer.
        Args:
            section: str, representing section unique name.

        Returns:
            tf.keras.layers.Dense
        """
        units = self.parser[section].get('units')
        if not units:
            assert (
                len(self.output_units) > self.output_count
            ), 'Output units given are less than dense layers required'
            units = self.output_units[self.output_count]
            self.output_count += 1
        activation = self.parser[section].get('activation')
        return Dense(
            units, activation, kernel_initializer=self.get_initializer(section)
        )

    def build_model(self):
        """
        Parse all configuration sections, create layers and model.

        Returns:
            tf.keras.Model
        """
        outputs = []
        common_layer = None
        input_layer = current_layer = Input(self.input_shape)
        for section in self.parser.sections():
            if section.startswith('convolutional'):
                current_layer = self.create_convolution(section)(current_layer)
            if section.startswith('flatten'):
                current_layer = Flatten()(current_layer)
            if section.startswith('dense'):
                current_layer = self.create_dense(section)(
                    common_layer if common_layer is not None else current_layer
                )
            if self.parser[section].get('common'):
                common_layer = current_layer
            if self.parser[section].get('output'):
                outputs.append(current_layer)
        self.output_count = 0
        model = Model(input_layer, outputs)
        if self.optimizer:
            model.compile(self.optimizer)
        return model
