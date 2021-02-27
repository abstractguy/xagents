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
        ema_decay=0.99,
        trust_region=True,
        *args,
        **kwargs,
    ):
        super(ACER, self).__init__(envs, model[0], n_steps=n_steps, *args, **kwargs)
        self.buffer_max_size = buffer_max_size
        self.buffers = [
            ReplayBuffer(buffer_max_size, batch_size=self.n_steps, seed=self.seed)
            for _ in range(self.n_envs)
        ]
        self.avg_model = model[1]
        self.replay_ratio = replay_ratio
        self.epsilon = epsilon
        self.delta = delta
        self.importance_c = importance_c
        self.batch_indices = tf.range(self.n_steps * self.n_envs, dtype=tf.int64)[
            :, tf.newaxis
        ]
        self.ema = tf.train.ExponentialMovingAverage(ema_decay)
        self.trust_region = trust_region

    @staticmethod
    def add_grads(g1, g2):
        if g1 is not None and g2 is not None:
            return g1 + g2
        if g1 is not None:
            return g1
        if g2 is not None:
            return g2

    def calculate_grads(
        self, tape, loss, predicted_action_probs, avg_action_probs, value_loss
    ):
        if not self.trust_region:
            return tape.gradient(loss, self.model.trainable_variables)
        g = tape.gradient(
            loss,
            predicted_action_probs,
        )
        k = -avg_action_probs / (predicted_action_probs + self.epsilon)
        adj = tf.maximum(
            0.0,
            (tf.reduce_sum(k * g, axis=-1) - self.delta)
            / (tf.reduce_sum(tf.square(k), axis=-1) + self.epsilon),
        )
        g = g - tf.reshape(adj, [self.n_envs * self.n_steps, 1]) * k
        grads_f = -g / (self.n_envs * self.n_steps)
        grads_policy = tape.gradient(
            predicted_action_probs, self.model.trainable_variables, grads_f
        )
        grads_q = tape.gradient(
            value_loss * self.value_loss_coef, self.model.trainable_variables
        )
        return [self.add_grads(g1, g2) for (g1, g2) in zip(grads_policy, grads_q)]

    def calculate_returns(
        self, values, rewards, dones, importance_bar, selected_critic_logits
    ):
        returns = []
        current_value = values[-1]
        for step in reversed(range(1, self.n_steps + 1)):
            i1 = step * self.n_envs
            i0 = i1 - self.n_envs
            returns.append(
                rewards[i0:i1] + self.gamma * current_value * (1 - dones[i0:i1])
            )
            current_value = (
                importance_bar[i0:i1] * (current_value - selected_critic_logits[i0:i1])
            ) + values[i0:i1]
        returns.reverse()
        return returns

    def calculate_loss(
        self,
        returns,
        values,
        log_probs,
        entropies,
        selected_ratios=None,
        selected_critic_logits=None,
    ):
        entropy = tf.reduce_mean(entropies)
        returns = tf.concat(returns, 0)
        advantage = returns - values
        action_gain = (
            log_probs * advantage * tf.minimum(self.importance_c, selected_ratios)
        )
        action_loss = -tf.reduce_mean(action_gain)
        value_loss = tf.reduce_mean(tf.square(returns - selected_critic_logits) * 0.5)
        if self.trust_region:
            return (
                -(action_loss - self.entropy_coef * entropy)
                * self.n_envs
                * self.n_steps
            ), value_loss
        return (
            action_loss
            + self.value_loss_coef * value_loss
            - self.entropy_coef * entropy
        ), value_loss

    def gradient_update(
        self, states, actions, action_probs, rewards, dones, log_probs, entropies
    ):
        with tf.GradientTape(True) as tape:
            avg_action_probs = self.avg_model(states)[-1]
            *_, critic_logits, _, predicted_action_probs = self.model(states)
            values = tf.reduce_sum(predicted_action_probs * critic_logits, axis=-1)
            action_indices = self.get_action_indices(self.batch_indices, actions)
            selected_critic_logits = tf.gather_nd(critic_logits, action_indices)
            importance_ratio = predicted_action_probs / (action_probs + self.epsilon)
            selected_ratios = tf.gather_nd(importance_ratio, action_indices)
            importance_bar = tf.minimum(1.0, selected_ratios)
            returns = self.calculate_returns(
                values, rewards, dones, importance_bar, selected_critic_logits
            )
            loss, value_loss = self.calculate_loss(
                returns,
                values,
                log_probs,
                entropies,
                selected_ratios,
                selected_critic_logits,
            )
        grads = self.calculate_grads(
            tape, loss, predicted_action_probs, avg_action_probs, value_loss
        )
        if self.grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.grad_norm)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    @tf.function
    def train_step(self):
        states, rewards, actions, values, dones, log_probs, entropies, action_probs = [
            tf.concat(item, 0) for item in self.get_batch()
        ]
        self.gradient_update(
            states, actions, action_probs, rewards, dones, log_probs, entropies
        )
        if (
            self.replay_ratio > 0
            and sum((len(buffer) for buffer in self.buffers)) >= self.buffer_max_size
        ):
            for _ in range(np.random.poisson(self.replay_ratio)):
                (
                    states,
                    actions,
                    rewards,
                    dones,
                    _,
                    action_probs,
                    log_probs,
                    entropies,
                ) = self.concat_buffer_samples()
                self.gradient_update(
                    states, actions, action_probs, rewards, dones, log_probs, entropies
                )


if __name__ == '__main__':
    from tensorflow.keras.optimizers import Adam

    from models import CNNA2C
    from utils import create_gym_env

    seed = None
    envi = create_gym_env('PongNoFrameskip-v4', 2)
    m1 = CNNA2C(
        envi[0].observation_space.shape,
        envi[0].action_space.n,
        critic_units=envi[0].action_space.n,
        seed=seed,
        actor_activation='softmax',
    )
    m2 = CNNA2C(
        envi[0].observation_space.shape,
        envi[0].action_space.n,
        critic_units=envi[0].action_space.n,
        seed=seed,
        actor_activation='softmax',
    )
    o = Adam(7e-4)
    agn = ACER(envi, (m1, m2), optimizer=o, seed=seed)
    agn.fit(19)
