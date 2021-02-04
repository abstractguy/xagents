import numpy as np
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
        advantage_epsilon=1e-5,
        clip_norm=0.1,
        *args,
        **kwargs,
    ):
        super(PPO, self).__init__(
            envs, model, transition_steps=transition_steps, *args, **kwargs
        )
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.mini_batches = mini_batches
        self.advantage_epsilon = advantage_epsilon
        self.clip_norm = clip_norm
        self.last_states = self.get_states()
        self.last_masks = np.ones(self.n_envs)

    def update_lasts(self, states, masks):
        self.last_states = states
        self.last_masks = masks
        return True

    @tf.function
    def train_step(self):
        transitions = states, rewards, values, returns, log_probs, actions, masks = [
            [] for _ in range(7)
        ]
        states.append(self.last_states)
        masks.append(self.last_masks)
        for step in range(self.transition_steps):
            step_actions, step_log_probs, _, step_values = self.model(states[step])
            step_states, step_rewards, step_dones = tf.numpy_function(
                func=self.step_envs,
                inp=[step_actions],
                Tout=(tf.float32, tf.float32, tf.float32),
            )
            step_masks = 1 - step_dones
            states.append(step_states)
            actions.append(step_actions)
            log_probs.append(step_log_probs)
            values.append(step_values)
            rewards.append(step_rewards)
            masks.append(step_masks)
        *_, next_values = self.model(states[-1])
        values.append(next_values)
        gae = 0
        for step in reversed(range(self.transition_steps)):
            delta = (
                rewards[step]
                + self.gamma * values[step + 1] * masks[step + 1]
                - values[step]
            )
            gae = delta + self.gamma * self.gae_lambda * masks[step + 1] * gae
            returns.append(gae + values[step])
        returns.reverse()
        tf.py_function(self.update_lasts, [states[-1], masks[-1]], tf.bool)
        states = tf.reshape(tf.concat(states[:-1], 0), (-1, *self.input_shape))
        rewards, values, returns, log_probs, actions, masks = [
            tf.reshape(tf.concat(item, 0), (-1, 1))
            if len(item) == self.transition_steps
            else tf.reshape(tf.concat(item[:-1], 0), (-1, 1))
            for item in transitions[1:]
        ]
        advantages = returns - values
        advantages = (advantages - tf.reduce_mean(advantages)) / (
            tf.math.reduce_std(advantages) + self.advantage_epsilon
        )
        batch_size = self.n_envs * self.transition_steps
        mini_batch_size = batch_size // self.mini_batches
        for epoch in range(self.ppo_epochs):
            indices = tf.random.shuffle(tf.range(batch_size))
            indices = tf.reshape(indices, (-1, mini_batch_size))
            for idx in indices:
                batches = (
                    states,
                    rewards,
                    values,
                    returns,
                    log_probs,
                    actions,
                    masks,
                    advantages,
                )
                (
                    states_mb,
                    rewards_mb,
                    values_mb,
                    returns_mb,
                    log_probs_mb,
                    actions_mb,
                    masks_mb,
                    advantages_mb,
                ) = [tf.gather(item, idx) for item in batches]
                with tf.GradientTape() as tape:
                    _, new_log_probs, entropy, new_values = self.model(
                        states_mb, actions=actions_mb
                    )
                    ratio = tf.exp(new_log_probs - log_probs_mb)
                    s1 = ratio * advantages_mb
                    s2 = (
                        tf.clip_by_value(ratio, 1 - self.clip_norm, 1 + self.clip_norm)
                        * advantages_mb
                    )
                    action_loss = tf.reduce_mean(-tf.minimum(s1, s2))
                    clipped_values = values_mb + tf.clip_by_value(
                        new_values - values_mb, -self.clip_norm, self.clip_norm
                    )
                    value_loss = tf.square(new_values - returns_mb)
                    clipped_value_loss = tf.square(clipped_values - returns_mb)
                    value_loss = 0.5 * tf.reduce_mean(
                        tf.maximum(value_loss, clipped_value_loss)
                    )
                    loss = (
                        value_loss * self.value_loss_coef
                        + action_loss
                        - entropy * self.entropy_coef
                    )
                gradients = tape.gradient(loss, self.model.trainable_variables)
                gradients, _ = tf.clip_by_global_norm(gradients, self.grad_norm)
                self.model.optimizer.apply_gradients(
                    zip(gradients, self.model.trainable_variables)
                )


if __name__ == '__main__':
    ens = create_gym_env('PongNoFrameskip-v4', 8)
    from tensorflow.keras.optimizers import Adam

    from models import CNNA2C

    m = CNNA2C(ens[0].observation_space.shape, ens[0].action_space.n)
    ac = PPO(ens, m, optimizer=Adam(25e-5, epsilon=1e-5))
    ac.fit(19)
