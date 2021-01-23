import os
from time import perf_counter, sleep

import cv2
import gym
import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras.optimizers import Adam

from base_agent import BaseAgent
from models import create_cnn_dqn
from utils import ReplayBuffer, create_gym_env


class DQN(BaseAgent):
    def __init__(
        self,
        envs,
        model,
        target_reward,
        buffer_max_size=10000,
        buffer_initial_size=None,
        buffer_batch_size=32,
        epsilon_start=1.0,
        epsilon_end=0.02,
        double=False,
        epsilon_decay_steps=150000,
        target_sync_steps=1000,
        *args,
        **kwargs,
    ):
        """
        Initialize agent settings.
        Args:
            envs: A list of gym environments that return states as atari frames.
            model: tf.keras.models.Model used for training.
            target_reward: A scalar when reached, training will stop.
            buffer_max_size: Replay buffer maximum size.
            buffer_initial_size: Replay buffer size to fill before training.
            buffer_batch_size: Batch size when any buffer from the given buffers
                get_sample() method is called.
            epsilon_start: Start value of epsilon that regulates exploration during training.
            epsilon_end: End value of epsilon which represents the minimum value of epsilon
                which will not be decayed further when reached.
            double: If True, DDQN is used for gradient updates.
            epsilon_decay_steps: Maximum steps that determine epsilon decay rate.
            target_sync_steps: Update target model every n steps.
            args: args that will be passed to BaseAgent()
            kwargs: kwargs that will be passed to BaseAgent()
        """
        super(DQN, self).__init__(envs, model, target_reward, *args, **kwargs)
        self.buffers = [
            ReplayBuffer(
                buffer_max_size,
                buffer_initial_size,
                self.transition_steps,
                self.gamma,
                buffer_batch_size,
            )
            for _ in range(self.n_envs)
        ]
        self.target_model = tf.keras.models.clone_model(self.model)
        self.buffer_batch_size = buffer_batch_size
        self.epsilon_start = self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.double = double
        self.batch_indices = tf.range(
            self.buffer_batch_size * self.n_envs, dtype=tf.int64
        )[:, tf.newaxis]
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_sync_steps = target_sync_steps

    def get_actions(self, training=True):
        """
        Generate action following an epsilon-greedy policy.
        Args:
            training: If False, no use of randomness will apply.
        Returns:
            A random action or Q argmax.
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.available_actions, self.n_envs)
        return self.model(np.array(self.states))[1]

    def get_action_indices(self, actions):
        """
        Get indices that will be passed to tf.gather_nd()
        Args:
            actions: Action tensor of shape self.batch_size
        Returns:
            Indices.
        """
        return tf.concat(
            (self.batch_indices, tf.cast(actions[:, tf.newaxis], tf.int64)), -1
        )

    @tf.function
    def get_targets(self, batch):
        """
        Get target values for gradient updates.
        Args:
            batch: A batch of observations in the form of
                [[states], [actions], [rewards], [dones], [next states]]
        Returns:
            None
        """
        states, actions, rewards, dones, new_states = batch
        q_states = self.model(states)[0]
        if self.double:
            new_state_actions = self.model(new_states)[1]
            new_state_q_values = self.target_model(new_states)[0]
            a = self.get_action_indices(new_state_actions)
            new_state_values = tf.gather_nd(new_state_q_values, a)
        else:
            new_state_values = tf.reduce_max(self.target_model(new_states)[0], axis=1)
        new_state_values = tf.where(
            dones, tf.constant(0, new_state_values.dtype), new_state_values
        )
        target_values = tf.identity(q_states)
        target_value_update = new_state_values * (
            self.gamma ** self.transition_steps
        ) + tf.cast(rewards, tf.float32)
        indices = self.get_action_indices(actions)
        target_values = tf.tensor_scatter_nd_update(
            target_values, indices, target_value_update
        )
        return target_values

    def fill_buffers(self):
        """
        Fill self.buffer up to its initial size.
        """
        total_size = sum(buffer.initial_size for buffer in self.buffers)
        sizes = {}
        for i, env in enumerate(self.envs):
            buffer = self.buffers[i]
            state = self.states[i]
            while len(buffer) < buffer.initial_size:
                action = env.action_space.sample()
                new_state, reward, done, _ = env.step(action)
                buffer.append((state, action, reward, done, new_state))
                state = new_state
                if done:
                    state = env.reset()
                sizes[i] = len(buffer)
                filled = sum(sizes.values())
                complete = round((filled / total_size) * 100, 2)
                print(
                    f'\rFilling replay buffer {i + 1}/{self.n_envs} ==> {complete}% | '
                    f'{filled}/{total_size}',
                    end='',
                )
        print()
        self.reset_envs()

    @tf.function
    def train_on_batch(self, x, y, sample_weight=None):
        """
        Train on a given batch.
        Args:
            x: States tensor
            y: Targets tensor
            sample_weight: sample_weight passed to model.compiled_loss()
        Returns:
            None
        """
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)[0]
            loss = self.model.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.model.losses
            )
        self.model.optimizer.minimize(loss, self.model.trainable_variables, tape=tape)
        self.model.compiled_metrics.update_state(y, y_pred, sample_weight)

    def fit(
        self,
        learning_rate=1e-4,
        monitor_session=None,
        weights=None,
    ):
        """
        Train agent on a supported environment
        Args:
            learning_rate: Model learning rate shared by both main and target networks.
            monitor_session: Session name to use for monitoring the training with wandb.
            weights: Path to .tf trained model weights to continue training.
            max_steps: Maximum number of steps, if reached the training will stop.
        Returns:
            None
        """
        if monitor_session:
            wandb.init(name=monitor_session)
        optimizer = Adam(learning_rate)
        if weights:
            self.model.load_weights(weights)
            self.target_model.load_weights(weights)
        self.model.compile(optimizer, loss='mse')
        self.target_model.compile(optimizer, loss='mse')
        self.fill_buffers()
        self.training_start_time = perf_counter()
        self.last_reset_time = self.training_start_time
        while True:
            self.last_reset_time = perf_counter()
            if self.training_done():
                break
            actions = self.get_actions()
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon_start - self.steps / self.epsilon_decay_steps,
            )
            training_batch = self.step_envs(actions)
            targets = self.get_targets(training_batch)
            self.train_on_batch(training_batch[0], targets)
            if self.steps % self.target_sync_steps == 0:
                self.target_model.set_weights(self.model.get_weights())

    def play(
        self,
        weights=None,
        video_dir=None,
        render=False,
        frame_dir=None,
        frame_delay=0.0,
        env_idx=0,
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
            action = self.get_actions(False)
            state, reward, done, _ = env_in_use.step(action)
            if done:
                break
            steps += 1
            sleep(frame_delay)


if __name__ == '__main__':
    gym_envs = create_gym_env('PongNoFrameskip-v4', 3)
    m = create_cnn_dqn(gym_envs[0].observation_space.shape, gym_envs[0].action_space.n)
    agn = DQN(gym_envs, m, 18)
    agn.fit(19)
