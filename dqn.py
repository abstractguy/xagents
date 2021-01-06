import os
from collections import defaultdict, deque
from time import perf_counter

import cv2
import gym
import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras.optimizers import Adam


class DQN:
    def __init__(
        self,
        envs,
        replay_buffers,
        models,
        buffer_batch_size=32,
        checkpoint=None,
        reward_buffer_size=100,
        epsilon_start=1.0,
        epsilon_end=0.02,
        n_steps=1,
        gamma=0.99,
        double=False,
    ):
        """
        Initialize agent settings.
        Args:
            envs: gym environment that returns states as atari frames.
            replay_buffers: ReplayBuffer object to use for memorizing transitions.
            models: Main and target models which should be of the same architecture.
            buffer_batch_size: Batch size when any buffer from the given buffers
                get_sample() method is called.
            checkpoint: Path to .tf filename under which the trained model will be saved.
            reward_buffer_size: Size of the reward buffer that will hold the last n total
                rewards which will be used for calculating the mean reward.
            epsilon_start: Start value of epsilon that regulates exploration during training.
            epsilon_end: End value of epsilon which represents the minimum value of epsilon
                which will not be decayed further when reached.
            n_steps: n-step transition for example given s1, s2, s3, s4 and n_step = 4,
                transition will be s1 -> s4 (defaults to 1, s1 -> s2)
            gamma: Discount factor used for gradient updates.
            double: If True, DDQN is used for gradient updates.
        """
        assert replay_buffers, 'No Replay buffers given'
        assert envs, 'No Environments given'
        self.n_envs = len(envs)
        assert len(replay_buffers) == self.n_envs, (
            f'Environments ({len(replay_buffers)}) '
            f'vs buffers mismatch ({self.n_envs})'
        )
        assert all(
            [buffer.n_steps == n_steps for buffer in replay_buffers]
        ), f'Buffer n-steps mismatch {self.buffers[0].n_steps}, {n_steps}'
        self.envs = envs
        self.env_ids = [id(env) for env in self.envs]
        self.buffers = {
            env_id: buffer for (env_id, buffer) in zip(self.env_ids, replay_buffers)
        }
        self.main_model, self.target_model = models
        self.buffer_batch_size = buffer_batch_size
        self.checkpoint_path = checkpoint
        self.total_rewards = deque(maxlen=reward_buffer_size * self.n_envs)
        self.best_reward = -float('inf')
        self.mean_reward = -float('inf')
        self.states = {}
        self.reset_envs()
        self.steps = 0
        self.frame_speed = 0
        self.last_reset_step = 0
        self.epsilon_start = self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.games = 0
        self.n_steps = n_steps
        self.gamma = gamma
        self.double = double
        self.batch_indices = tf.range(
            self.buffer_batch_size * self.n_envs, dtype=tf.int64
        )[:, tf.newaxis]
        self.episode_rewards = defaultdict(lambda: 0)

    def reset_envs(self):
        for env_id, env in zip(self.env_ids, self.envs):
            self.states[env_id] = env.reset()

    def get_action(self, state, training=True):
        """
        Generate action following an epsilon-greedy policy.
        Args:
            state: Atari frame that needs an action.
            training: If False, no use of randomness will apply.
        Returns:
            A random action or Q argmax.
        """
        if training and np.random.random() < self.epsilon:
            return self.envs[0].action_space.sample()
        q_values = self.main_model(np.expand_dims(state, 0)).numpy()
        return np.argmax(q_values)

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
        q_states = self.main_model(states)
        if self.double:
            new_state_actions = tf.argmax(self.main_model(new_states), 1)
            new_state_q_values = self.target_model(new_states)
            a = self.get_action_indices(new_state_actions)
            new_state_values = tf.gather_nd(new_state_q_values, a)
        else:
            new_state_values = tf.reduce_max(self.target_model(new_states), axis=1)
        new_state_values = tf.where(
            dones, tf.constant(0, new_state_values.dtype), new_state_values
        )
        target_values = tf.identity(q_states)
        target_value_update = new_state_values * (self.gamma ** self.n_steps) + tf.cast(
            rewards, tf.float32
        )
        indices = self.get_action_indices(actions)
        target_values = tf.tensor_scatter_nd_update(
            target_values, indices, target_value_update
        )
        return target_values

    def checkpoint(self):
        """
        Save model weights if current reward > best reward.
        Returns:
            None
        """
        if self.best_reward < self.mean_reward:
            print(f'Best reward updated: {self.best_reward} -> {self.mean_reward}')
            if self.checkpoint_path:
                self.main_model.save_weights(self.checkpoint_path)
        self.best_reward = max(self.mean_reward, self.best_reward)

    def display_metrics(self):
        """
        Display progress metrics to the console.
        Returns:
            None
        """
        display_titles = (
            'frame',
            'games',
            'speed',
            'mean reward',
            'best reward',
            'epsilon',
            'episode rewards',
        )
        display_values = (
            self.steps,
            self.games,
            f'{round(self.frame_speed)} steps/s',
            self.mean_reward,
            self.best_reward,
            np.around(self.epsilon, 2),
            list(self.total_rewards)[-self.n_envs :],
        )
        display = (
            f'{title}: {value}' for title, value in zip(display_titles, display_values)
        )
        print(', '.join(display))

    def update_metrics(self, start_time):
        """
        Update progress metrics.
        Args:
            start_time: Episode start time, used for calculating fps.
        Returns:
            None
        """
        self.checkpoint()
        self.frame_speed = (self.steps - self.last_reset_step) / (
            perf_counter() - start_time
        )
        self.last_reset_step = self.steps
        self.mean_reward = np.around(np.mean(self.total_rewards), 2)

    def fill_buffers(self):
        """
        Fill self.buffer up to its initial size.
        """
        total_size = sum(buffer.initial_size for buffer in self.buffers.values())
        sizes = {}
        for i, env in enumerate(self.envs, 1):
            env_id = id(env)
            buffer = self.buffers[env_id]
            state = self.states[env_id]
            while len(buffer) < buffer.initial_size:
                action = env.action_space.sample()
                new_state, reward, done, _ = env.step(action)
                buffer.append((state, action, reward, done, new_state))
                state = new_state
                if done:
                    state = env.reset()
                sizes[env_id] = len(buffer)
                filled = sum(sizes.values())
                complete = round((filled / total_size) * 100, 2)
                print(
                    f'\rFilling replay buffer {i}/{self.n_envs} ==> {complete}% | '
                    f'{filled}/{total_size}',
                    end='',
                )
        print()
        self.reset_envs()

    @tf.function
    def train_on_batch(self, model, x, y, sample_weight=None):
        """
        Train on a given batch.
        Args:
            model: tf.keras.Model
            x: States tensor
            y: Targets tensor
            sample_weight: sample_weight passed to model.compiled_loss()

        Returns:
            None
        """
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = model.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=model.losses
            )
        model.optimizer.minimize(loss, model.trainable_variables, tape=tape)
        model.compiled_metrics.update_state(y, y_pred, sample_weight)

    def get_training_batch(self, done_envs):
        batches = []
        for env_id, env in zip(self.env_ids, self.envs):
            state = self.states[env_id]
            action = self.get_action(state)
            buffer = self.buffers[env_id]
            new_state, reward, done, _ = env.step(action)
            self.steps += 1
            self.episode_rewards[env_id] += reward
            buffer.append((state, action, reward, done, new_state))
            self.states[env_id] = new_state
            if done:
                done_envs.append(1)
                self.total_rewards.append(self.episode_rewards[env_id])
                self.games += 1
                self.episode_rewards[env_id] = 0
                self.states[env_id] = env.reset()
            batch = buffer.get_sample()
            batches.append(batch)
        if len(batches) > 1:
            return [np.concatenate(item) for item in zip(*batches)]
        return batches[0]

    def fit(
        self,
        target_reward,
        decay_n_steps=150000,
        learning_rate=1e-4,
        update_target_steps=1000,
        monitor_session=None,
        weights=None,
        max_steps=None,
    ):
        """
        Train agent on a supported environment
        Args:
            target_reward: Target reward, if achieved, the training will stop
            decay_n_steps: Maximum steps that determine epsilon decay rate.
            learning_rate: Model learning rate shared by both main and target networks.
            update_target_steps: Update target model every n steps.
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
            self.main_model.load_weights(weights)
            self.target_model.load_weights(weights)
        self.main_model.compile(optimizer, loss='mse')
        self.target_model.compile(optimizer, loss='mse')
        self.fill_buffers()
        done_envs = []
        start_time = perf_counter()
        while True:
            if len(done_envs) == self.n_envs:
                self.update_metrics(start_time)
                start_time = perf_counter()
                self.display_metrics()
                done_envs.clear()
            if self.mean_reward >= target_reward:
                print(f'Reward achieved in {self.steps} steps!')
                break
            if max_steps and self.steps >= max_steps:
                print(f'Maximum steps exceeded')
                break
            self.epsilon = max(
                self.epsilon_end, self.epsilon_start - self.steps / decay_n_steps
            )
            training_batch = self.get_training_batch(done_envs)
            targets = self.get_targets(training_batch)
            self.train_on_batch(self.main_model, training_batch[0], targets)
            if self.steps % update_target_steps == 0:
                self.target_model.set_weights(self.main_model.get_weights())

    def play(self, weights=None, video_dir=None, render=False, frame_dir=None):
        """
        Play and display a game.
        Args:
            weights: Path to trained weights, if not specified, the most recent
                model weights will be used.
            video_dir: Path to directory to save the resulting game video.
            render: If True, the game will be displayed.
            frame_dir: Path to directory to save game frames.
        Returns:
            None
        """
        if weights:
            self.main_model.load_weights(weights)
        if video_dir:
            self.envs = gym.wrappers.Monitor(self.envs, video_dir)
        self.states = self.envs.reset()
        steps = 0
        for dir_name in (video_dir, frame_dir):
            os.makedirs(dir_name or '.', exist_ok=True)
        while True:
            if render:
                self.envs.render()
            if frame_dir:
                frame = self.envs.render(mode='rgb_array')
                cv2.imwrite(os.path.join(frame_dir, f'{steps:05d}.jpg'), frame)
            action = self.get_action(False)
            self.states, reward, done, _ = self.envs.step(action)
            if done:
                break
            steps += 1


if __name__ == '__main__':
    from models import dqn_conv
    from utils import ReplayBuffer, create_gym_env

    nnn = 10000
    ppp = True
    nss = 3
    bf = [
        ReplayBuffer(nnn, n_steps=nss),
        ReplayBuffer(nnn, n_steps=nss),
        ReplayBuffer(nnn, n_steps=nss),
    ]
    en = [
        create_gym_env('PongNoFrameskip-v4'),
        create_gym_env('PongNoFrameskip-v4'),
        create_gym_env('PongNoFrameskip-v4'),
    ]
    mod1 = dqn_conv(en[0].observation_space.shape, en[0].action_space.n, duel=True)
    mod2 = dqn_conv(en[0].observation_space.shape, en[0].action_space.n, duel=True)
    agn = DQN(en, bf, (mod1, mod2), n_steps=nss, double=True)
    agn.fit(19)
    # agn.play(
    #     '/Users/emadboctor/Desktop/code/dqn-pong-19-model/pong_test.tf', render=True
    # )
