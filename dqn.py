import numpy as np
import tensorflow as tf

from base_agent import BaseAgent
from utils import ReplayBuffer, create_gym_env


class DQN(BaseAgent):
    def __init__(
        self,
        envs,
        model,
        buffer_max_size=10000,
        buffer_initial_size=None,
        buffer_batch_size=32,
        epsilon_start=1.0,
        epsilon_end=0.02,
        double=False,
        update_target_steps=1000,
        decay_n_steps=150000,
        *args,
        **kwargs,
    ):
        """
        Initialize DQN agent.
        Args:
            envs: A list of gym environments.
            model: tf.keras.models.Model
            buffer_max_size: Maximum size for each replay buffer used by its
                respective environment.
            buffer_initial_size: Initial replay buffer size, if not specified,
                buffer_max_size is used
            buffer_batch_size: Sample size returned by each replay buffer sample() call.
            epsilon_start: Epsilon value at training start.
            epsilon_end: Epsilon value at training end.
            double: If True, DDQN is used.
            *args: args Passed to BaseAgent
            **kwargs: kwargs Passed to BaseAgent
        """
        super(DQN, self).__init__(envs, model, *args, **kwargs)
        self.buffers = [
            ReplayBuffer(
                buffer_max_size,
                buffer_initial_size,
                self.n_steps,
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
        self.update_target_steps = update_target_steps
        self.decay_n_steps = decay_n_steps
        self.batch_indices = tf.range(
            self.buffer_batch_size * self.n_envs, dtype=tf.int64
        )[:, tf.newaxis]

    def get_actions(self):
        """
        Generate action following an epsilon-greedy policy.
        Returns:
            A random action or Q argmax.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.available_actions, self.n_envs)
        return self.model(self.get_states())[0]

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

    def get_targets(self, batch):
        """
        Get target values for gradient updates.
        Args:
            batch: A batch of observations in the form of
                [[states], [actions], [rewards], [dones], [next states]]
        Returns:
            target values used as y_true in gradient update.
        """
        states, actions, rewards, dones, new_states = batch
        q_states = self.model(states)[1]
        if self.double:
            new_state_actions = self.model(new_states)[0]
            new_state_q_values = self.target_model(new_states)[1]
            a = self.get_action_indices(new_state_actions)
            new_state_values = tf.gather_nd(new_state_q_values, a)
        else:
            new_state_values = tf.reduce_max(self.target_model(new_states)[1], axis=1)
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
                complete = round((filled / total_size) * 100, self.metric_digits)
                print(
                    f'\rFilling replay buffer {i + 1}/{self.n_envs} ==> {complete}% | '
                    f'{filled}/{total_size}',
                    end='',
                )
        print()
        self.reset_envs()

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
            y_pred = self.model(x, training=True)[1]
            loss = self.model.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.model.losses
            )
        self.model.optimizer.minimize(loss, self.model.trainable_variables, tape=tape)
        self.model.compiled_metrics.update_state(y, y_pred, sample_weight)

    def get_training_batch(self):
        """
        Join batches sampled from each environment in self.envs
        Returns:
            batch: A batch of observations in the form of
                [[states], [actions], [rewards], [dones], [next states]]
        """
        batches = []
        for i, env in enumerate(self.envs):
            buffer = self.buffers[i]
            batch = buffer.get_sample()
            batches.append(batch)
        if len(batches) > 1:
            joined = [np.concatenate(item).astype(np.float32) for item in zip(*batches)]
        else:
            joined = [item.astype(np.float32) for item in batches[0]]
        joined[-2] = joined[-2].astype(np.bool)
        return joined

    def at_step_start(self):
        self.epsilon = max(
            self.epsilon_end, self.epsilon_start - self.steps / self.decay_n_steps
        )

    def at_step_end(self):
        if self.steps % self.update_target_steps == 0:
            self.target_model.set_weights(self.model.get_weights())

    @tf.function
    def train_step(self):
        """
        Do 1 training step.
        Returns:
            None
        """
        actions = tf.numpy_function(self.get_actions, [], tf.int64)
        tf.numpy_function(self.step_envs, [actions], (tf.float32, tf.float32, tf.bool))
        training_batch = tf.numpy_function(
            self.get_training_batch,
            [],
            (tf.float32, tf.float32, tf.float32, tf.bool, tf.float32),
        )
        targets = self.get_targets(training_batch)
        self.train_on_batch(training_batch[0], targets)


if __name__ == '__main__':
    gym_envs = create_gym_env('PongNoFrameskip-v4', 3)
    from tensorflow.keras.optimizers import Adam

    from models import create_cnn_dqn

    m = create_cnn_dqn(gym_envs[0].observation_space.shape, gym_envs[0].action_space.n)
    agn = DQN(gym_envs, m, optimizer=Adam(1e-4), buffer_max_size=10000)
    agn.fit(18)
    # agn.play(
    #     '/Users/emadboctor/Desktop/code/drl-models/dqn-pong-19-model/pong_test.tf',
    #     render=True,
    # )
