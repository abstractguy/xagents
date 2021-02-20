import numpy as np
import tensorflow as tf

from a2c import A2C
from utils import ReplayBuffer


class ACER(A2C):
    def __init__(
        self,
        envs,
        model,
        n_steps=20,
        buffer_max_size=10000,
        replay_ratio=4,
        epsilon=1e-6,
        delta=1,
        importance_c=10.0,
        *args,
        **kwargs,
    ):
        super(ACER, self).__init__(envs, model, n_steps=n_steps, *args, **kwargs)
        self.buffers = [
            ReplayBuffer(buffer_max_size, batch_size=self.n_envs * self.n_steps)
            for _ in range(self.n_envs)
        ]
        self.replay_ratio = replay_ratio
        self.epsilon = epsilon
        self.delta = delta
        self.importance_c = importance_c
        self.batch_indices = tf.range(self.n_steps * self.n_envs, dtype=tf.int64)[
            :, tf.newaxis
        ]

    def calculate_returns(
        self, rewards, values, dones, selected_logits, action_importance
    ):
        returns = []
        last_return = values[-self.n_envs :]
        action_importance = tf.minimum(1.0, action_importance)
        for step in reversed(range(self.n_steps)):
            returns.append(rewards[step] + self.gamma * last_return * (1 - dones[step]))
            last_return = (
                action_importance[step] * (last_return - selected_logits[step])
            ) + values[step]
        returns.reverse()
        return np.asarray(returns).reshape(-1)

    def compute_loss(
        self,
        returns,
        values,
        entropies,
        action_importance,
        value_logits,
        importance_ratio,
        action_probs,
        selected_probs,
        selected_logits,
    ):
        entropy = tf.reduce_mean(entropies)
        advantage = returns - values
        selected_log_probs = tf.math.log(selected_probs) + self.epsilon
        action_gain = (
            selected_log_probs
            * advantage
            * tf.minimum(self.importance_c, action_importance)
        )
        action_loss = -tf.reduce_mean(action_gain)
        advantage_bias_correction = value_logits - tf.expand_dims(values, -1)
        log_probs_bias_correction = tf.expand_dims(
            tf.math.log(selected_probs + self.epsilon), -1
        )
        gain_bias_correction = tf.reduce_sum(
            log_probs_bias_correction
            * advantage_bias_correction
            * tf.nn.relu(1.0 - (self.importance_c / (importance_ratio + self.epsilon)))
            * action_probs,
            1,
        )
        loss_bias_correction = -tf.reduce_mean(gain_bias_correction)
        value_loss = tf.reduce_mean(0.5 * tf.square(returns - selected_logits))
        return (
            action_loss
            + loss_bias_correction
            + value_loss * self.value_loss_coef
            - entropy * self.entropy_coef
        )

    @tf.function
    def train_step(self):
        with tf.GradientTape() as tape:
            (
                states,
                rewards,
                actions,
                value_logits,
                dones,
                log_probs,
                entropies,
                actor_logits,
            ) = self.get_batch()
            actions, actor_logits, value_logits, log_probs = [
                tf.concat(item, 0)
                for item in [actions, actor_logits, value_logits, log_probs]
            ]
            action_probs = tf.nn.softmax(actor_logits)
            values = tf.reduce_sum(action_probs * value_logits, axis=-1)
            action_indices = self.get_action_indices(self.batch_indices, actions)
            selected_probs = tf.gather_nd(action_probs, action_indices)
            selected_logits = tf.gather_nd(value_logits, action_indices)
            importance_ratio = action_probs / (action_probs + self.epsilon)
            action_importance = tf.gather_nd(importance_ratio, action_indices)
            returns = tf.numpy_function(
                self.calculate_returns,
                [rewards, values, dones, selected_logits, action_importance],
                tf.float32,
            )
            loss = self.compute_loss(
                returns,
                values,
                entropies,
                action_importance,
                value_logits,
                importance_ratio,
                action_probs,
                selected_probs,
                selected_logits,
            )
        grads = tape.gradient(loss, self.model.trainable_variables)
        if self.grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.grad_norm)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


if __name__ == '__main__':
    from tensorflow.keras.optimizers import Adam
    from tensorflow_addons.optimizers import MovingAverage

    from models import CNNA2C
    from utils import create_gym_env

    envi = create_gym_env('PongNoFrameskip-v4', 16)
    m = CNNA2C(
        envi[0].observation_space.shape,
        envi[0].action_space.n,
        critic_units=envi[0].action_space.n,
    )
    o = MovingAverage(Adam(7e-4))
    agn = ACER(envi, m, optimizer=o)
    agn.fit(19)
