import random
from collections import deque

import numpy as np


class BaseBuffer:
    """
    Base class for replay buffers.
    """

    def __init__(self, size, initial_size=None, batch_size=32):
        """
        Initialize replay buffer.
        Args:
            size: Buffer maximum size.
            initial_size: Buffer initial size to be filled before training starts.
                To be used by caller.
            batch_size: Size of the batch that should be used in get_sample() implementation.
        """
        if initial_size:
            assert size >= initial_size, 'Buffer initial size exceeds max size'
        self.size = size
        self.initial_size = initial_size or size
        self.batch_size = batch_size
        self.current_size = 0

    def append(self, *args):
        """
        Add experience to buffer.
        Args:
            *args: Items to store, types are implementation specific.

        """
        raise NotImplementedError(
            f'append() should be implemented by {self.__class__.__name__} subclasses'
        )

    def get_sample(self):
        """
        Sample from stored experience.

        Returns:
            Sample as numpy array.
        """
        raise NotImplementedError(
            f'get_sample() should be implemented by {self.__class__.__name__} subclasses'
        )


class ReplayBuffer(BaseBuffer):
    """
    deque-based replay buffer that holds state transitions
    """

    def __init__(self, size, n_steps=1, gamma=0.99, **kwargs):
        """
        Initialize replay buffer.
        Args:
            size: Buffer maximum size.
            n_steps: Steps separating start and end states.
            gamma: Discount factor.
            **kwargs: kwargs passed to BaseBuffer.
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
        Sample from stored experience.

        Returns:
            Same number of args passed to append, having self.batch_size as
            first shape.
        """
        memories = random.sample(self.main_buffer, self.batch_size)
        if self.batch_size > 1:
            return [np.array(item) for item in zip(*memories)]
        return memories[0]


class IAmTheOtherKindOfReplayBufferBecauseFuckTensorflow(BaseBuffer):
    """
    numpy-based replay buffer added for compatibility with tensorflow shortcomings.
    """

    def __init__(self, size, slots, **kwargs):
        """
        Initialize replay buffer.

        Args:
            size: Buffer maximum size.
            slots: Number of args that will be passed to self.append()
            **kwargs: kwargs passed to BaseBuffer.
        """
        super(IAmTheOtherKindOfReplayBufferBecauseFuckTensorflow, self).__init__(
            size, **kwargs
        )
        self.slots = [np.array([])] * slots
        self.current_size = 0

    def append(self, *args):
        """
        Add experience to buffer.
        Args:
            *args: Items to store.

        Returns:
            None
        """
        self.current_size += 1
        for i, arg in enumerate(args):
            if not isinstance(arg, np.ndarray):
                arg = np.array([arg])
            if not self.slots[i].shape[0]:
                self.slots[i] = np.zeros((self.size, *arg.shape), np.float32)
            self.slots[i][self.current_size % self.size] = arg.copy()

    def get_sample(self):
        """
        Sample from stored experience.

        Returns:
            Same number of args passed to append, having self.batch_size as
            first shape.
        """
        indices = np.random.randint(
            0, min(self.current_size, self.size), self.batch_size
        )
        return [slot[indices] for slot in self.slots]