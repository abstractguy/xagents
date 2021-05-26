import numpy as np
import tensorflow as tf
from gym.spaces.discrete import Discrete

from base_agents import OffPolicy


class DQN(OffPolicy):
    def __init__(
        self,
        envs,
        model,
        double=False,
        **kwargs,
    ):
        """
        Initialize DQN agent.
        Args:
            envs: A list of gym environments.
            model: tf.keras.models.Model that is expected to be compiled
                with an optimizer before training starts.
            buffer_max_size: Maximum size for each replay buffer used by its
                respective environment.
            buffer_initial_size: Initial replay buffer size, if not specified,
                buffer_max_size is used
            buffer_batch_size: Sample size returned by each replay buffer sample() call.
            epsilon_start: Epsilon value at training start.
            epsilon_end: Epsilon value at training end.
            double: If True, DDQN is used.
            target_sync_steps: Update target network every n steps.
            epsilon_decay_steps: Decay epsilon for n steps.
            **kwargs: kwargs Passed to OnPolicy
        """
        super(DQN, self).__init__(envs, model, **kwargs)
        assert isinstance(envs[0].action_space, Discrete), (
            f'Invalid environment: {envs[0].spec.id}. DQN supports '
            f'environments with a discrete action space only, got {envs[0].action_space}'
        )
        self.target_model = tf.keras.models.clone_model(self.model)
        self.double = double
        self.batch_indices = tf.range(
            self.buffer_batch_size * self.n_envs, dtype=tf.int64
        )[:, tf.newaxis]

    @tf.function
    def get_model_outputs(self, inputs, models, training=True):
        """
        Get inputs and apply normalization if `scale_factor` was specified earlier,
        then return model outputs.
        Args:
            inputs: Inputs as tensors / numpy arrays that are expected
                by the given model.
            models: A tf.keras.Model
            training: `training` parameter passed to model call.

        Returns:
            Outputs that is expected from the given model.
        """
        q_values = super(DQN, self).get_model_outputs(inputs, models, training)
        return tf.argmax(q_values, 1), q_values

    def sync_target_model(self):
        if self.steps % self.target_sync_steps == 0:
            self.target_model.set_weights(self.model.get_weights())

    def get_actions(self):
        """
        Generate action following an epsilon-greedy policy.

        Returns:
            A random action or Q argmax.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions, self.n_envs)
        return self.get_model_outputs(self.get_states(), self.output_models)[0]

    def get_targets(self, states, actions, rewards, dones, new_states):
        """
        Get targets for gradient update.
        Args:
            states: A tensor of shape (self.n_envs * self.buffer_batch_size, *self.input_shape)
            actions: A tensor of shape (self.n_envs * self.buffer_batch_size)
            rewards: A tensor of shape (self.n_envs * self.buffer_batch_size)
            dones: A tensor of shape (self.n_envs * self.buffer_batch_size)
            new_states: A tensor of shape (self.n_envs * self.buffer_batch_size, *self.input_shape)

        Returns:
            Target values, a tensor of shape (self.n_envs * self.buffer_batch_size, self.n_actions)
        """
        q_states = self.get_model_outputs(states, self.model)[1]
        if self.double:
            new_state_actions = self.get_model_outputs(new_states, self.model)[0]
            new_state_q_values = self.get_model_outputs(new_states, self.target_model)[
                1
            ]
            a = self.get_action_indices(self.batch_indices, new_state_actions)
            new_state_values = tf.gather_nd(new_state_q_values, a)
        else:
            new_state_values = tf.reduce_max(
                self.get_model_outputs(new_states, self.target_model)[1], axis=1
            )
        new_state_values = tf.where(
            tf.cast(dones, tf.bool),
            tf.constant(0, new_state_values.dtype),
            new_state_values,
        )
        target_values = tf.identity(q_states)
        target_value_update = new_state_values * (self.gamma ** self.n_steps) + tf.cast(
            rewards, tf.float32
        )
        indices = self.get_action_indices(self.batch_indices, actions)
        target_values = tf.tensor_scatter_nd_update(
            target_values, indices, target_value_update
        )
        return target_values

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
            y_pred = self.get_model_outputs(x, self.model)[1]
            loss = tf.reduce_mean(tf.square(y - y_pred))
        self.model.optimizer.minimize(loss, self.model.trainable_variables, tape=tape)
        self.model.compiled_metrics.update_state(y, y_pred, sample_weight)

    def at_step_start(self):
        """
        Execute steps that will run before self.train_step() which decays epsilon.

        Returns:
            None
        """
        self.update_epsilon()

    @tf.function
    def train_step(self):
        """
        Perform 1 step which controls action_selection, interaction with environments
        in self.envs, batching and gradient updates.

        Returns:
            None
        """
        actions = self.get_actions()
        tf.numpy_function(self.step_envs, [actions, False, True], [])
        training_batch = tf.numpy_function(
            self.concat_buffer_samples,
            [],
            5 * [tf.float32],
        )
        targets = self.get_targets(*training_batch)
        self.train_on_batch(training_batch[0], targets)

    def at_step_end(self):
        """
        Execute steps that will run after self.train_step() which
        updates target model.

        Returns:
            None
        """
        self.sync_target_model()


if __name__ == '__main__':
    from utils import ModelHandler, create_gym_env

    gym_envs = create_gym_env('PongNoFrameskip-v4', 3)
    seed = None
    from tensorflow.keras.optimizers import Adam

    optimizer = Adam(1e-4)
    mh = ModelHandler(
        'models/cnn/dqn.cfg', [6], gym_envs[0].observation_space.shape, optimizer, seed
    )
    m = mh.build_model()
    agn = DQN(gym_envs, m, buffer_max_size=10000, seed=seed)
    agn.fit(19)
    # agn.play(
    #     '/Users/emadboctor/Desktop/code/models-drl/dqn-pong-19-model/pong_test.tf',
    #     render=True,
    # )
