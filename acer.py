import numpy as np
import tensorflow as tf

from a2c import A2C
from utils import ReplayBuffer


class ACER(A2C):
    def __init__(
        self,
        envs,
        models,
        n_steps=20,
        grad_norm=10,
        buffer_max_size=100000,
        buffer_initial_size=None,
        replay_ratio=4,
        epsilon=1e-6,
        delta=1,
        importance_c=10.0,
        ema_decay=0.99,
        trust_region=True,
        *args,
        **kwargs,
    ):
        super(ACER, self).__init__(
            envs, models[0], n_steps=n_steps, grad_norm=grad_norm, *args, **kwargs
        )
        if replay_ratio > 0:
            self.buffers = [
                ReplayBuffer(
                    buffer_max_size // self.n_envs,
                    initial_size=buffer_initial_size,
                    batch_size=1,
                    seed=self.seed,
                )
                for _ in range(self.n_envs)
            ]
        self.avg_model = models[1]
        self.replay_ratio = replay_ratio
        self.epsilon = epsilon
        self.delta = delta
        self.importance_c = importance_c
        self.ema = tf.train.ExponentialMovingAverage(ema_decay)
        self.batch_indices = tf.range(self.n_steps * self.n_envs, dtype=tf.int64)[
            :, tf.newaxis
        ]
        self.trust_region = trust_region

    def store_batch(self, batch):
        buffer_items = [[] for _ in range(self.n_envs)]
        for batch_item in batch:
            env_results = np.swapaxes(batch_item, 0, 1)
            for i, buffer_item in enumerate(buffer_items):
                buffer_item.append(env_results[i])
        for i, buffer in enumerate(self.buffers):
            buffer.append(buffer_items[i])

    def get_batch(self):
        states, rewards, actions, _, dones, *_, action_probs = super(
            ACER, self
        ).get_batch()
        states.append(self.get_states())
        dones = dones[1:]
        batch = [
            np.asarray(item, np.float32)
            for item in [states, rewards, actions, dones, action_probs]
        ]
        self.store_batch(batch)
        return self.concat_step_batches(*batch)

    def calculate_returns(
        self, rewards, dones, selected_logits, values, selected_importance
    ):
        importance_bar = self.flat_to_steps(tf.minimum(1.0, selected_importance))
        dones = self.flat_to_steps(dones, self.n_steps)
        rewards = self.flat_to_steps(rewards)
        selected_logits = self.flat_to_steps(selected_logits)
        values = self.flat_to_steps(values, self.n_steps + 1)
        current_return = values[-1]
        returns = []
        for step in reversed(range(self.n_steps)):
            current_return = rewards[step] + self.gamma * current_return * (
                1.0 - dones[step]
            )
            returns.append(current_return)
            current_return = (
                importance_bar[step] * (current_return - selected_logits[step])
            ) + values[step]
        return tf.reshape(tf.stack(returns[::-1], 1), [-1])

    @staticmethod
    def add_grads(g1, g2):
        assert not (g1 is None and g2 is None), 'Both gradients to add are None'
        if g1 is not None and g2 is not None:
            return g1 + g2
        if g1 is not None:
            return g1
        return g2

    def calculate_loss(
        self,
        returns,
        values,
        entropies,
        selected_probs=None,
        selected_importance=None,
        selected_logits=None,
    ):
        entropy = tf.reduce_mean(entropies)
        values = values[: -self.n_envs]
        advantage = returns - values
        log_probs = tf.math.log(selected_probs + self.epsilon)
        action_gain = log_probs * tf.stop_gradient(
            advantage * tf.minimum(self.importance_c, selected_importance)
        )
        action_loss = -tf.reduce_mean(action_gain)
        value_loss = (
            tf.reduce_mean(
                input_tensor=tf.square(tf.stop_gradient(returns) - selected_logits)
                * 0.5
            )
            * self.value_loss_coef
        )
        if self.trust_region:
            return (
                -(action_loss - self.entropy_coef * entropy)
                * self.n_steps
                * self.n_envs
            ), value_loss
        return action_loss + value_loss * value_loss - self.entropy_coef * entropy

    def calculate_grads(self, tape, loss, action_probs, avg_action_probs, value_loss):
        if not self.trust_region:
            return tape.gradient(loss, self.model.trainable_variables)
        g = tape.gradient(
            loss,
            action_probs,
        )
        k = -avg_action_probs / (action_probs + self.epsilon)
        adj = tf.maximum(
            0.0,
            (tf.reduce_sum(input_tensor=k * g, axis=-1) - self.delta)
            / (tf.reduce_sum(input_tensor=tf.square(k), axis=-1) + self.epsilon),
        )
        g = g - tf.reshape(adj, [self.n_envs * self.n_steps, 1]) * k
        grads_f = -g / (self.n_envs * self.n_steps)
        grads_policy = tape.gradient(
            action_probs, self.model.trainable_variables, grads_f
        )
        grads_q = tape.gradient(value_loss, self.model.trainable_variables)
        return [self.add_grads(g1, g2) for (g1, g2) in zip(grads_policy, grads_q)]

    def gradient_update(self, states, rewards, actions, dones, step_action_probs):
        action_indices = self.get_action_indices(self.batch_indices, actions)
        with tf.GradientTape(persistent=True) as tape:
            *_, critic_logits, entropies, action_probs = self.model(states)
            *_, avg_critic_logits, _, avg_action_probs = self.avg_model(states)
            values = tf.reduce_sum(action_probs * critic_logits, axis=-1)
            action_probs, avg_action_probs, critic_logits = [
                tf.squeeze(item[: -self.n_envs])
                for item in [action_probs, avg_action_probs, critic_logits]
            ]
            selected_probs = tf.gather_nd(action_probs, action_indices)
            selected_logits = tf.gather_nd(critic_logits, action_indices)
            importance_weights = action_probs / (step_action_probs + self.epsilon)
            selected_importance = tf.gather_nd(importance_weights, action_indices)
            returns = self.calculate_returns(
                rewards, dones, selected_logits, values, selected_importance
            )
            loss = self.calculate_loss(
                returns,
                values,
                entropies,
                selected_probs,
                selected_importance,
                selected_logits,
            )
            value_loss = loss[1] if isinstance(loss, tuple) else None
        grads = self.calculate_grads(
            tape, loss, action_probs, avg_action_probs, value_loss
        )
        if self.grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.grad_norm)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.ema.apply(self.model.trainable_variables)
        tf.numpy_function(self.update_avg_weights, [], [])

    # @tf.function
    def train_step(self):
        numpy_func_dtypes = [tf.float32 for _ in range(5)]
        batch = tf.numpy_function(self.get_batch, [], numpy_func_dtypes)
        self.gradient_update(*batch)
        if (
            self.replay_ratio > 0
            and len(self.buffers[0]) == self.buffers[0].initial_size
        ):
            for _ in range(np.random.poisson(self.replay_ratio)):
                batch = tf.numpy_function(
                    self.concat_buffer_samples, [], numpy_func_dtypes
                )
                self.gradient_update(*batch)

    def update_avg_weights(self):
        avg_variables = [
            self.ema.average(weight).numpy()
            for weight in self.model.trainable_variables
        ]
        self.avg_model.set_weights(avg_variables)


if __name__ == '__main__':
    from tensorflow.keras.optimizers import Adam

    from models import CNNA2C
    from utils import create_gym_env

    seed = None
    envi = create_gym_env('PongNoFrameskip-v4', 2)
    ms = [
        CNNA2C(
            envi[0].observation_space.shape,
            envi[0].action_space.n,
            critic_units=envi[0].action_space.n,
            seed=seed,
            actor_activation='softmax',
        )
        for _ in range(2)
    ]
    o = Adam()
    agn = ACER(envi, ms, optimizer=o, seed=seed, buffer_initial_size=500)
    agn.fit(19)
