import numpy as np
import tensorflow as tf

from a2c import A2C


class PPO(A2C):
    def __init__(
        self,
        envs,
        model,
        n_steps=128,
        lam=0.95,
        ppo_epochs=4,
        mini_batches=4,
        advantage_epsilon=1e-8,
        clip_norm=0.1,
        **kwargs,
    ):
        """
        Initialize PPO agent.
        Args:
            envs: A list of gym environments.
            model: tf.keras.models.Model used for training.
            n_steps: n-step transition for example given s1, s2, s3, s4 and n_step = 4,
                transition will be s1 -> s4 (defaults to 1, s1 -> s2)
            lam: GAE-Lambda for advantage estimation
            ppo_epochs: Gradient updates per training step.
            mini_batches: Number of mini batches to use per gradient update.
            advantage_epsilon: Epsilon value added to estimated advantage.
            clip_norm: Clipping value passed to tf.clip_by_value()
            **kwargs: kwargs Passed to BaseAgent
        """
        super(PPO, self).__init__(envs, model, n_steps=n_steps, **kwargs)
        self.lam = lam
        self.ppo_epochs = ppo_epochs
        self.mini_batches = mini_batches
        self.advantage_epsilon = advantage_epsilon
        self.clip_norm = clip_norm
        self.batch_size = self.n_envs * self.n_steps
        self.mini_batch_size = self.batch_size // self.mini_batches

    def calculate_returns(self, states, rewards, values, dones):
        """
        Get a batch of returns.
        Args:
            states: states as numpy array of shape (self.n_steps, self.n_envs, *self.input_shape)
            rewards: rewards as numpy array of shape (self.n_steps, self.n_envs)
            values: values as numpy array of shape (self.n_steps, self.n_envs)
            dones: dones as numpy array of shape (self.n_steps, self.n_envs)

        Returns:
            returns as numpy array.
        """
        next_values = self.get_model_outputs(states[-1], self.output_models)[2].numpy()
        next_values = (
            np.expand_dims(next_values, 0)
            if not len(next_values.shape)
            else next_values
        )
        returns = []
        last_lam = 0
        values = np.concatenate([values, np.expand_dims(next_values, 0)])
        dones = np.concatenate([dones, np.expand_dims(dones[-1], 0)])
        for step in reversed(range(self.n_steps)):
            next_non_terminal = 1 - dones[step + 1]
            next_values = values[step + 1]
            delta = (
                rewards[step]
                + self.gamma * next_values * next_non_terminal
                - values[step]
            )
            last_lam = delta + self.gamma * self.lam * next_non_terminal * last_lam
            returns.append(last_lam)
        return np.asarray(returns[::-1]) + values[:-1]

    def update_gradients(
        self, states, actions, old_values, returns, old_log_probs, advantages
    ):
        """
        Perform gradient updates.
        Args:
            states: states as numpy array of shape (self.mini_batch_size, *self.input_shape)
            actions: actions as numpy array of shape (self.mini_batch_size,)
            old_values: old values as numpy array of shape (self.mini_batch_size,)
            returns: returns as numpy array of shape (self.mini_batch_size,)
            old_log_probs: old log probs as numpy array of shape (self.mini_batch_size,)
            advantages: advantages as numpy array of shape (self.mini_batch_size,)

        Returns:
            None
        """
        with tf.GradientTape() as tape:
            _, log_probs, values, entropy, _ = self.get_model_outputs(
                states, self.output_models, actions=actions
            )
            entropy = tf.reduce_mean(entropy)
            clipped_values = old_values + tf.clip_by_value(
                values - old_values, -self.clip_norm, self.clip_norm
            )
            value_loss1 = tf.square(values - returns)
            value_loss2 = tf.square(clipped_values - returns)
            value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_loss1, value_loss2))
            ratio = tf.exp(log_probs - old_log_probs)
            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * tf.clip_by_value(
                ratio, 1 - self.clip_norm, 1 + self.clip_norm
            )
            pg_loss = tf.reduce_mean(tf.maximum(pg_loss1, pg_loss2))
            loss = (
                pg_loss
                - entropy * self.entropy_coef
                + value_loss * self.value_loss_coef
            )
        grads = tape.gradient(loss, self.model.trainable_variables)
        if self.grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.grad_norm)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def get_mini_batches(self, *args):
        mini_batches = []
        indices = tf.range(self.batch_size)
        for _ in range(self.ppo_epochs):
            indices = tf.random.shuffle(indices)
            for i in range(0, self.batch_size, self.mini_batch_size):
                batch_indices = indices[i : i + self.mini_batch_size]
                mini_batches.append([tf.gather(item, batch_indices) for item in args])
        return mini_batches

    def run_ppo_epochs(self, states, actions, returns, old_values, old_log_probs):
        """
        Split batch into mini batches and perform gradient updates.
        Args:
            states: states as numpy array of shape (self.n_steps * self.n_envs, *self.input_shape)
            actions: actions as numpy array of shape (self.n_steps * self.n_envs,)
            returns: returns as numpy array of shape (self.n_steps * self.n_envs,)
            old_values: old values as numpy array of shape (self.n_steps * self.n_envs,)
            old_log_probs: old log probs as numpy array of shape (self.n_steps * self.n_envs,)

        Returns:
            None
        """
        for mini_batch in self.get_mini_batches(
            states, actions, returns, old_values, old_log_probs
        ):
            (
                states_mb,
                actions_mb,
                returns_mb,
                old_values_mb,
                old_log_probs_mb,
            ) = mini_batch
            advantages_mb = returns_mb - old_values_mb
            advantages_mb = (advantages_mb - tf.reduce_mean(advantages_mb)) / (
                tf.math.reduce_std(advantages_mb) + self.advantage_epsilon
            )
            self.update_gradients(
                states_mb,
                actions_mb,
                old_values_mb,
                returns_mb,
                old_log_probs_mb,
                advantages_mb,
            )

    def get_batch(self):
        (
            states,
            rewards,
            actions,
            values,
            dones,
            log_probs,
            *_,
        ) = [np.asarray(item, np.float32) for item in super(PPO, self).get_batch()]
        values = np.expand_dims(values, -1) if len(values.shape) <= 1 else values
        returns = self.calculate_returns(states, rewards, values, dones)
        return self.concat_step_batches(states, actions, returns, values, log_probs)

    @tf.function
    def train_step(self):
        """
        Perform 1 step which controls action_selection, interaction with environments
        in self.envs, batching and gradient updates.

        Returns:
            None
        """
        batch = tf.numpy_function(self.get_batch, [], 5 * [tf.float32])
        self.run_ppo_epochs(*batch)


if __name__ == '__main__':
    from tensorflow.keras.optimizers import Adam

    from utils import ModelHandler, create_gym_env

    envi = create_gym_env('PongNoFrameskip-v4', 16)
    mh = ModelHandler('models/cnn-ac.cfg', [envi[0].action_space.n, 1])
    m = mh.build_model()
    agn = PPO(envi, m, optimizer=Adam(25e-5))
    agn.fit(19)
