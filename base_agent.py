import os
from collections import deque
from datetime import timedelta
from time import perf_counter, sleep

import cv2
import gym
import numpy as np
import wandb
from tensorflow.keras.optimizers import Adam


class BaseAgent:
    def __init__(
        self,
        envs,
        model,
        optimizer=None,
        checkpoint=None,
        reward_buffer_size=100,
        n_steps=1,
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
            n_steps: n-step transition for example given s1, s2, s3, s4 and n_step = 4,
                transition will be s1 -> s4 (defaults to 1, s1 -> s2)
            gamma: Discount factor used for gradient updates.
            metric_digits: Rounding decimals for display purposes.
        """
        assert envs, 'No Environments given'
        self.n_envs = len(envs)
        self.envs = envs
        self.model = model
        self.optimizer = optimizer or Adam()
        self.checkpoint_path = checkpoint
        self.total_rewards = deque(maxlen=reward_buffer_size)
        self.n_steps = n_steps
        self.gamma = gamma
        self.metric_digits = metric_digits
        self.target_reward = None
        self.max_steps = None
        self.input_shape = self.envs[0].observation_space.shape
        self.available_actions = self.envs[0].action_space.n
        self.best_reward = -float('inf')
        self.mean_reward = -float('inf')
        self.states = [None] * self.n_envs
        self.dones = [False] * self.n_envs
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
        if self.mean_reward > self.best_reward:
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
            'time',
            'steps',
            'games',
            'speed',
            'mean reward',
            'best reward',
            'episode rewards',
        )
        display_values = (
            timedelta(seconds=perf_counter() - self.training_start_time),
            self.steps,
            self.games,
            f'{round(self.frame_speed)} steps/s',
            self.mean_reward,
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
        if len(self.done_envs) >= self.n_envs:
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
            self.dones[i] = done
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

    def init_training(self, target_reward, max_steps, monitor_session, weights, loss):
        """
        Initialize training start time, wandb session & models (self.model / self.target_model)
        Args:
            target_reward: Scalar values whenever achieved, the training will stop.
            max_steps: Maximum steps, if exceeded, the training will stop.
            monitor_session: Wandb session name.
            weights: Path to .tf weights file compatible with self.model
            loss: loss passed to tf.keras.models.Model.compile(loss=loss)

        Returns:
            None
        """
        self.target_reward = target_reward
        self.max_steps = max_steps
        if monitor_session:
            wandb.init(name=monitor_session)
        if weights:
            self.model.load_weights(weights)
            if hasattr(self, 'target_model'):
                self.target_model.load_weights(weights)
        self.model.compile(self.optimizer, loss=loss)
        if hasattr(self, 'target_model'):
            self.target_model.compile(self.optimizer, loss=loss)
        self.training_start_time = perf_counter()
        self.last_reset_time = perf_counter()

    def train_step(self):
        raise NotImplementedError(
            'train_step() should be implemented by BaseAgent subclasses'
        )

    def at_step_start(self):
        pass

    def at_step_end(self):
        pass

    def get_states(self):
        return np.array(self.states, np.float32)

    def get_dones(self):
        return np.array(self.dones, np.float32)

    def fit(
        self,
        target_reward,
        max_steps=None,
        monitor_session=None,
        weights=None,
    ):
        """
        Train DQN agent on a supported environment.
        Args:
            target_reward: Target reward, if achieved, the training will stop
            max_steps: Maximum number of steps, if reached the training will stop.
            monitor_session: Session name to use for monitoring the training with wandb.
            weights: Path to .tf trained model weights to continue training.
        Returns:
            None
        """
        if hasattr(self, 'fill_buffers'):
            self.fill_buffers()
        self.init_training(
            target_reward,
            max_steps,
            monitor_session,
            weights,
            'mse',
        )
        while True:
            self.check_episodes()
            if self.training_done():
                break
            self.at_step_start()
            self.train_step()
            self.at_step_end()

    def play(
        self,
        weights=None,
        video_dir=None,
        render=False,
        frame_dir=None,
        frame_delay=0.0,
        env_idx=0,
        action_idx=0,
    ):
        """
        Play and display a game.
        Args:
            weights: Path to trained weights, if not specified, the most recent
                model weights will be used.
            video_dir: Path to directory to save the resulting game video.
            render: If True, the game will be displayed.
            frame_dir: Path to directory to save game frames.
            frame_delay: Delay between rendered frames.
            env_idx: env index in self.envs
            action_idx: Index of action output by self.model
        Returns:
            None
        """
        self.reset_envs()
        env_in_use = self.envs[env_idx]
        if weights:
            self.model.load_weights(weights)
        if video_dir:
            env_in_use = gym.wrappers.Monitor(env_in_use, video_dir)
        steps = 0
        for dir_name in (video_dir, frame_dir):
            os.makedirs(dir_name or '.', exist_ok=True)
        while True:
            if render:
                env_in_use.render()
            if frame_dir:
                frame = env_in_use.render(mode='rgb_array')
                cv2.imwrite(os.path.join(frame_dir, f'{steps:05d}.jpg'), frame)
            action = self.model(self.get_states())[action_idx][env_idx]
            self.states[env_idx], reward, done, _ = env_in_use.step(action)
            if done:
                break
            steps += 1
            sleep(frame_delay)
