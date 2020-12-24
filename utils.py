import cv2
import gym
import numpy as np


class AtariPreprocessor(gym.Wrapper):
    def __init__(self, env, skips=4, frame_shape=(84, 84)):
        super(AtariPreprocessor, self).__init__(env)
        self.skips = skips
        self.frame_shape = frame_shape
        self.observation_space.shape = (*frame_shape, 1)

    def process_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, self.frame_shape) / 255
        return np.expand_dims(frame, -1)

    def step(self, action):
        total_reward = 0
        state, done, info = 3 * [None]
        for _ in range(self.skips):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return self.process_frame(state), total_reward, done, info

    def reset(self, **kwargs):
        return self.process_frame(self.env.reset())
