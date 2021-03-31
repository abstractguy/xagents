import os
import random
from collections import deque
from datetime import timedelta
from time import perf_counter, sleep

import cv2
import gym
import numpy as np
import tensorflow as tf
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
        custom_loss=None,
        seed=None,
        scale_inputs=False,
    ):
        """
        Base class for various types of agents.
        Args:
            envs: A list of gym environments.
            model: tf.keras.models.Model used for training.
            optimizer: tf.keras.optimizers.Optimizer.
            checkpoint: Path to .tf filename under which the trained model will be saved.
            reward_buffer_size: Size of the reward buffer that will hold the last n total
                rewards which will be used for calculating the mean reward.
            n_steps: n-step transition for example given s1, s2, s3, s4 and n_step = 4,
                transition will be s1 -> s4 (defaults to 1, s1 -> s2)
            gamma: Discount factor used for gradient updates.
            metric_digits: Rounding decimals for display purposes.
            custom_loss: Loss passed to tf.keras.models.Model.compile()
            seed: Random seed passed to random.seed(), np.random.seed(), tf.random.seed(),
                env.seed()
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
        self.custom_loss = custom_loss
        self.seed = seed
        self.scale_inputs = scale_inputs
        self.target_reward = None
        self.max_steps = None
        self.input_shape = self.envs[0].observation_space.shape
        self.n_actions = self.envs[0].action_space.n
        self.best_reward = -float('inf')
        self.mean_reward = -float('inf')
        self.states = [None] * self.n_envs
        self.dones = [False] * self.n_envs
        self.steps = 0
        self.updates = 0
        self.frame_speed = 0
        self.last_reset_step = 0
        self.training_start_time = None
        self.last_reset_time = None
        self.games = 0
        self.episode_rewards = np.zeros(self.n_envs)
        self.episode_scores = deque(maxlen=self.n_envs)
        self.done_envs = []
        if seed:
            tf.random.set_seed(seed)
            np.random.seed(seed)
            for env in self.envs:
                env.seed(seed)
            os.environ['PYTHONHASHSEED'] = f'{seed}'
            random.seed(seed)
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
        Check whether a target reward or maximum number of steps is reached.

        Returns:
            bool
        """
        if self.mean_reward >= self.target_reward:
            print(f'Reward achieved in {self.steps} steps')
            return True
        if self.max_steps and self.steps >= self.max_steps:
            print(f'Maximum steps exceeded')
            return True
        return False

    def concat_buffer_samples(self, dtypes=None):
        """
        Concatenate samples drawn from each environment respective buffer.
        Args:
            dtypes: A list of respective numpy dtypes to return.

        Returns:
            A list of concatenated samples.
        """
        if hasattr(self, 'buffers'):
            batches = []
            for i, env in enumerate(self.envs):
                buffer = self.buffers[i]
                batch = buffer.get_sample()
                batches.append(batch)
            dtypes = dtypes or [np.float32 for _ in range(len(batches[0]))]
            if len(batches) > 1:
                return [
                    np.concatenate(item).astype(dtype)
                    for (item, dtype) in zip(zip(*batches), dtypes)
                ]
            return [item.astype(dtype) for (item, dtype) in zip(batches[0], dtypes)]

    def step_envs(self, actions, get_observation=False, store_in_buffers=False):
        """
        Step environments in self.envs, update metrics (if any done games)
            and return / store results.
        Args:
            actions: A list / numpy array of actions to execute by environments.
            get_observation: If True, a list of [states, actions, rewards, dones, new_states]
                of length self.n_envs each will be returned.
            store_in_buffers: If True, each observation is saved separately in respective buffer.

        Returns:
            A list of observations as numpy arrays or an empty list.
        """
        observations = []
        for (
            (i, env),
            action,
            *items,
        ) in zip(enumerate(self.envs), actions):
            state = self.states[i]
            new_state, reward, done, _ = env.step(action)
            self.states[i] = new_state
            self.dones[i] = done
            self.steps += 1
            self.episode_rewards[i] += reward
            observation = state, action, reward, done, new_state
            if store_in_buffers and hasattr(self, 'buffers'):
                self.buffers[i].append(observation)
            if get_observation:
                observations.append(observation)
            if done:
                self.done_envs.append(1)
                self.total_rewards.append(self.episode_rewards[i])
                self.episode_scores.append(self.episode_rewards[i])
                self.games += 1
                self.episode_rewards[i] = 0
                self.states[i] = env.reset()
        return [np.array(item, np.float32) for item in zip(*observations)]

    def init_training(self, target_reward, max_steps, monitor_session, weights):
        """
        Initialize training start time, wandb session & models (self.model / self.target_model)
        Args:
            target_reward: Scalar values whenever achieved, the training will stop.
            max_steps: Maximum steps, if exceeded, the training will stop.
            monitor_session: Wandb session name.
            weights: Path to .tf weights file compatible with self.model

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
        self.model.compile(self.optimizer, loss=self.custom_loss)
        if hasattr(self, 'target_model'):
            self.target_model.compile(self.optimizer, loss=self.custom_loss)
        self.training_start_time = perf_counter()
        self.last_reset_time = perf_counter()

    def train_step(self):
        """
        Perform 1 step which controls action_selection, interaction with environments
        in self.envs, batching and gradient updates.

        Returns:
            None
        """
        raise NotImplementedError(
            'train_step() should be implemented by BaseAgent subclasses'
        )

    def get_model_outputs(self, inputs, model, training=True):
        raise NotImplementedError(
            'get_model_outputs() should be implemented by BaseAgent subclasses'
        )

    def at_step_start(self):
        """
        Execute steps that will run before self.train_step().

        Returns:
            None
        """
        pass

    def at_step_end(self):
        """
        Execute steps that will run after self.train_step().

        Returns:
            None
        """
        pass

    def get_states(self):
        """
        Get most recent states.

        Returns:
            self.states as numpy array.
        """
        return np.array(self.states, np.float32)

    def get_dones(self):
        """
        Get most recent game statuses.

        Returns:
            self.dones as numpy array.
        """
        return np.array(self.dones, np.float32)

    @staticmethod
    def get_action_indices(batch_indices, actions):
        """
        Get indices that will be passed to tf.gather_nd()
        Args:
            batch_indices: tf.range() result of the same shape as the batch size.
            actions: Action tensor of same shape as the batch size.

        Returns:
            Indices as tf tensors.
        """
        return tf.concat((batch_indices, tf.cast(actions[:, tf.newaxis], tf.int64)), -1)

    @staticmethod
    def concat_step_batches(*args):
        """
        Concatenate n-step batches.
        Args:
            *args: A list of numpy arrays which will be concatenated separately.

        Returns:
            A list of concatenated numpy arrays.
        """
        return [a.swapaxes(0, 1).reshape(-1, *a.shape[2:]) for a in args]

    def get_model_inputs(self, inputs):
        if not self.scale_inputs:
            return inputs
        return tf.cast(inputs, tf.float32) / 255.0

    def fit(
        self,
        target_reward,
        max_steps=None,
        monitor_session=None,
        weights=None,
    ):
        """
        Common training loop shared by subclasses, monitors training status
        and progress, performs all training steps, updates metrics, and logs progress.
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
        self.init_training(target_reward, max_steps, monitor_session, weights)
        while True:
            self.check_episodes()
            if self.training_done():
                break
            self.at_step_start()
            self.train_step()
            self.at_step_end()
            self.updates += 1

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
            self.model.load_weights(weights).expect_partial()
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
            action = self.get_model_outputs(self.get_states(), self.model, False)[
                action_idx
            ][env_idx]
            self.states[env_idx], reward, done, _ = env_in_use.step(action)
            if done:
                break
            steps += 1
            sleep(frame_delay)
