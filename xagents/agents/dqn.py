import numpy as np
import tensorflow as tf
from xagents.agents.base_agents import OffPolicy
from gym.spaces.discrete import Discrete


class DQN(OffPolicy):
    def __init__(
        self,
        envs,
        model,
        buffers,
        double=False,
        **kwargs,
    ):
        """
        Initialize DQN agent.
        Args:
            envs: A list of gym environments.
            model: tf.keras.models.Model that is expected to be compiled
                with an optimizer before training starts.
            buffers: A list of replay buffer objects whose length should match
                `envs`s'.
            double: If True, DDQN is used.
            **kwargs: kwargs passed to super classes.
        """
        super(DQN, self).__init__(envs, model, buffers, **kwargs)
        assert isinstance(envs[0].action_space, Discrete), (
            f'Invalid environment: {envs[0].spec.id}. DQN supports '
            f'environments with a discrete action space only, got {envs[0].action_space}'
        )
        self.target_model = tf.keras.models.clone_model(self.model)
        self.double = double
        self.batch_indices = tf.range(
            self.buffers[0].batch_size * self.n_envs, dtype=tf.int64
        )[:, tf.newaxis]

    @tf.function
    def get_model_outputs(self, inputs, models, training=True):
        """
        Get inputs and apply normalization if `scale_factor` was specified earlier,
        then return model outputs.
        Args:
            inputs: Inputs as tensors / numpy arrays that are expected
                by the given model.
            models: A tf.keras.Model or a list of tf.keras.Model(s)
            training: `training` parameter passed to model call.

        Returns:
            Outputs that is expected from the given model.
        """
        q_values = super(DQN, self).get_model_outputs(inputs, models, training)
        return tf.argmax(q_values, 1), q_values

    def sync_target_model(self):
        """
        Sync target model weights with main's

        Returns:
            None
        """
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
            states: A tensor of shape (self.n_envs * total buffer batch size, *self.input_shape)
            actions: A tensor of shape (self.n_envs * total buffer batch size)
            rewards: A tensor of shape (self.n_envs * total buffer batch size)
            dones: A tensor of shape (self.n_envs * total buffer batch size)
            new_states: A tensor of shape (self.n_envs * total buffer batch size, *self.input_shape)

        Returns:
            Target values, a tensor of shape (self.n_envs * total buffer batch size, self.n_actions)
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
        actions = tf.numpy_function(self.get_actions, [], tf.int64)
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
    from xagents.utils import ModelReader, ReplayBuffer, create_gym_env

    en = create_gym_env('PongNoFrameskip-v4', 3)
    seed = None
    from tensorflow.keras.optimizers import Adam

    optimizer = Adam(1e-4)
    mh = ModelReader(
        '../models/cnn/dqn.cfg', [6], en[0].observation_space.shape, optimizer, seed
    )
    m = mh.build_model()
    bs = [ReplayBuffer(10000 // len(en)) for _ in range(len(en))]
    agn = DQN(en, m, bs, seed=seed)
    agn.fit(19)
    # agn.play(
    #     '/Users/emadboctor/Desktop/code/models-drl/dqn-pong-19-model/pong_test.tf',
    #     render=True,
    # )
