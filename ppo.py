import tensorflow as tf

from a2c import A2C
from utils import create_gym_env


class PPO(A2C):
    def __init__(
        self,
        envs,
        model,
        transition_steps=128,
        gae_lambda=0.95,
        ppo_epochs=4,
        mini_batches=4,
        *args,
        **kwargs,
    ):
        """
        Initialize PPO agent.
        Args:
            envs: A list of gym environments.
            model: tf.keras.models.Model
            *args: args Passed to BaseAgent.
            **kwargs: kwargs Passed to BaseAgent.
        """
        super(PPO, self).__init__(
            envs, model, transition_steps=transition_steps, *args, **kwargs
        )
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.mini_batches = mini_batches

    def calculate_returns(
        self, states, actions, masks, rewards, values, log_probs, entropies
    ):
        """
        Calculate returns to be used for loss calculation and gradient update.
        Args:
            masks: Empty list that will be the same size as self.transition_steps
                and will contain done masks.
            rewards: Empty list that will be the same size as self.transition_steps
                and will contain self.step_envs() rewards.
            values: Empty list that will be the same size as self.transition_steps
                and will contain self.step_envs() values.
            log_probs: Empty list that will be the same size as self.transition_steps
                and will contain self.step_envs() log_probs.
            entropies: Empty list that will be the same size as self.transition_steps
                and will contain self.step_envs() entropies.

        Returns:
            returns (a list that has most recent values after performing n steps)
        """
        self.step_transitions(
            states, actions, log_probs, values, rewards, masks, entropies
        )
        next_values = self.model(states[-1])[-1]
        values.append(next_values)
        gae = 0
        returns = []
        for step in reversed(range(self.transition_steps)):
            delta = (
                rewards[step]
                + self.gamma * values[step + 1] * masks[step]
                - values[step]
            )
            gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    @tf.function
    def train_step(self):
        """
        Do 1 training step.

        Returns:
            None
        """
        states = []
        actions = []
        masks = []
        rewards = []
        values = []
        log_probs = []
        entropies = []
        returns = self.calculate_returns(
            states, actions, masks, rewards, values, log_probs, entropies
        )
        tf_returns = tf.concat(returns, axis=0)
        tf_observations = tf.concat(states[:-1], axis=0)
        tf_actions = tf.concat(actions, axis=0)
        tf_old_log_probs = tf.concat(log_probs, axis=0)
        tf_values = tf.concat(values[:-1], axis=0)
        tf_adv_target = tf_returns - tf_values
        tf_adv_target = (tf_adv_target - tf.reduce_mean(tf_adv_target)) / (
            tf.math.reduce_std(tf_adv_target) + 1e-5
        )
        batch_size = self.n_envs * self.transition_steps
        mini_batch_size = batch_size // self.mini_batches
        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        entropy_loss_epoch = 0.0
        for _ in range(self.ppo_epochs):
            indx = tf.random.shuffle(tf.range(batch_size))
            indx = tf.reshape(indx, (-1, mini_batch_size))
            for sample in indx:
                obs_batch = tf.gather(tf_observations, sample)
                returns_batch = tf.gather(tf_returns, sample)
                adv_target_batch = tf.gather(tf_adv_target, sample)
                action_batch = tf.gather(tf_actions, sample)
                old_log_probs_batch = tf.gather(tf_old_log_probs, sample)
                values_batch = tf.gather(tf_values, sample)
                with tf.GradientTape() as tape:
                    _, action_log_probs, dist_entropy, value = self.model(
                        obs_batch, action_batch
                    )
                    ratio = tf.exp(action_log_probs - old_log_probs_batch)
                    surr1 = -ratio * adv_target_batch
                    surr2 = (
                        -tf.clip_by_value(
                            ratio, 1.0 - self.clip_norm, 1.0 + self.clip_norm
                        )
                        * adv_target_batch
                    )
                    action_loss = tf.reduce_mean(tf.maximum(surr1, surr2))
                    value_pred_clipped = values_batch + tf.clip_by_value(
                        value - values_batch, -self.clip_norm, self.clip_norm
                    )
                    value_losses = tf.square(value - returns_batch)
                    value_losses_clipped = tf.square(value_pred_clipped - returns_batch)
                    value_loss = 0.5 * tf.reduce_mean(
                        tf.maximum(value_losses, value_losses_clipped)
                    )
                    entropy_loss = tf.reduce_mean(dist_entropy)
                    loss = (
                        self.value_loss_coef * value_loss
                        + action_loss
                        - entropy_loss * self.entropy_coef
                    )
                gradients = tape.gradient(loss, self.model.trainable_variables)
                gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
                self.model.optimizer.apply_gradients(
                    zip(gradients, self.model.trainable_variables)
                )
                value_loss_epoch += value_loss
                action_loss_epoch += action_loss
                entropy_loss_epoch += entropy_loss


if __name__ == '__main__':
    ens = create_gym_env('PongNoFrameskip-v4', 16)
    from tensorflow.keras.optimizers import Adam

    from models import CNNA2C

    m = CNNA2C(ens[0].observation_space.shape, ens[0].action_space.n)
    ac = PPO(ens, m, optimizer=Adam(25e-5, epsilon=1e-5))
    ac.fit(19)
