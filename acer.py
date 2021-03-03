import numpy as np
import tensorflow as tf

from a2c import A2C
from utils import ReplayBuffer


def batch_to_seq(h, nbatch, nsteps, flat=False):
    if flat:
        h = tf.reshape(h, [nbatch, nsteps])
    else:
        h = tf.reshape(h, [nbatch, nsteps, -1])
    return [
        tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=nsteps, value=h)
    ]


def seq_to_batch(h, nh=None):
    if nh:
        return tf.reshape(tf.concat(axis=1, values=h), [-1, nh])
    else:
        return tf.reshape(tf.stack(values=h, axis=1), [-1])


def strip(var, nenvs, nsteps, flat=False, nh=None):
    items = batch_to_seq(var, nenvs, nsteps + 1, flat)
    return seq_to_batch(items[:-1], nh)


class ACER(A2C):
    def __init__(
        self,
        envs,
        models,
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
        super(ACER, self).__init__(envs, models[0], n_steps=n_steps, *args, **kwargs)
        if replay_ratio > 0:
            self.buffers = [
                ReplayBuffer(
                    buffer_max_size // self.n_envs, batch_size=1, seed=self.seed
                )
                for _ in range(self.n_envs)
            ]
        self.avg_model = models[1]
        self.avg_model.set_weights(self.model.get_weights())
        self.replay_ratio = replay_ratio
        self.epsilon = epsilon
        self.delta = delta
        self.importance_c = importance_c
        self.ema_decay = ema_decay
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
        batch = [
            np.asarray(item, np.float32)
            for item in [states, rewards, actions, dones, action_probs]
        ]
        self.store_batch(batch)
        return self.concat_step_batches(*batch)

    @staticmethod
    def add_grads(g1, g2):
        if g1 is not None and g2 is not None:
            return g1 + g2
        if g1 is not None:
            return g1
        return g2

    def calculate_returns(
        self, rewards, dones, selected_logits, selected_ratios, values
    ):
        returns = []
        importance_bar = batch_to_seq(
            tf.minimum(1.0, selected_ratios), self.n_envs, self.n_steps, True
        )
        rewards, dones, selected_logits = [
            batch_to_seq(item, self.n_envs, self.n_steps, True)
            for item in [rewards, dones[self.n_envs :], selected_logits]
        ]
        values = batch_to_seq(values, self.n_envs, self.n_steps + 1, True)
        next_values = values[-1]
        for step in reversed(range(self.n_steps)):
            returns.append(rewards[step] + self.gamma * next_values * (1 - dones[step]))
            next_values = (
                importance_bar[step] * (next_values - selected_logits[step])
            ) + values[step]
        return seq_to_batch(returns[::-1])

    def calculate_loss(
        self,
        returns,
        values,
        entropies,
        *args,
        log_probs=None,
    ):
        selected_probs, selected_ratios, selected_logits = args
        entropy = tf.reduce_mean(entropies)
        values = strip(values, self.n_envs, self.n_steps, True)
        advantages = returns - values
        log_probs = tf.math.log(selected_probs + self.epsilon)
        action_gain = log_probs * tf.stop_gradient(
            advantages * tf.minimum(self.importance_c, selected_ratios)
        )
        action_loss = -tf.reduce_mean(action_gain)
        value_loss = tf.reduce_mean(
            tf.square(tf.stop_gradient(returns) - selected_logits) * 0.5
        )
        if self.trust_region:
            return (
                -(action_loss - entropy * self.entropy_coef)
                * self.n_envs
                * self.n_steps
            ), value_loss
        return (
            action_loss
            - entropy * self.entropy_coef
            + value_loss * self.value_loss_coef
        )

    def calculate_grads(
        self, tape, losses, new_action_probs, avg_action_probs, action_probs
    ):
        if not self.trust_region:
            return tape.gradient(losses, self.model.trainable_variables)
        g = tape.gradient(
            losses[0],
            new_action_probs,
        )
        k = -avg_action_probs / (action_probs + self.epsilon)
        adj = tf.maximum(
            0.0,
            (tf.reduce_sum(k * g, axis=-1) - self.delta)
            / (tf.reduce_sum(tf.square(k), axis=-1) + self.epsilon),
        )
        g = g - tf.reshape(adj, [self.n_envs * self.n_steps, 1]) * k
        grads_f = -g / (self.n_envs * self.n_steps)
        grads_policy = tape.gradient(
            new_action_probs, self.model.trainable_variables, grads_f
        )
        grads_q = tape.gradient(
            losses[1] * self.value_loss_coef, self.model.trainable_variables
        )
        return [self.add_grads(g1, g2) for (g1, g2) in zip(grads_policy, grads_q)]

    def gradient_update(self, states, rewards, actions, dones, action_probs):
        with tf.GradientTape(persistent=True) as tape:
            (
                *_,
                critic_logits,
                entropies,
                new_action_probs,
            ) = self.model(states)
            avg_action_probs = self.avg_model(states)[-1]
            values = tf.reduce_sum(new_action_probs * critic_logits, axis=-1)
            new_action_probs, avg_action_probs, critic_logits = [
                strip(item, self.n_envs, self.n_steps, nh=self.available_actions)
                for item in [new_action_probs, avg_action_probs, critic_logits]
            ]
            action_indices = self.get_action_indices(self.batch_indices, actions)
            selected_probs = tf.gather_nd(new_action_probs, action_indices)
            selected_logits = tf.gather_nd(critic_logits, action_indices)
            importance_ratios = new_action_probs / (action_probs + self.epsilon)
            selected_ratios = tf.gather_nd(importance_ratios, action_indices)
            returns = self.calculate_returns(
                rewards, dones, selected_logits, selected_ratios, values
            )
            losses = self.calculate_loss(
                returns,
                values,
                entropies,
                selected_probs,
                selected_ratios,
                selected_logits,
            )
        grads = self.calculate_grads(
            tape, losses, new_action_probs, avg_action_probs, action_probs
        )
        if self.grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.grad_norm)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    @tf.function
    def train_step(self):
        numpy_func_output = [tf.float32 for _ in range(5)]
        batch = tf.numpy_function(self.get_batch, [], numpy_func_output)
        self.gradient_update(*batch)
        if (
            self.replay_ratio > 0
            and len(self.buffers[0]) == self.buffers[0].initial_size
        ):
            for _ in range(np.random.poisson(self.replay_ratio)):
                buffer_samples = [buffer.get_sample() for buffer in self.buffers]
                batch = tf.numpy_function(
                    self.concat_step_batches, buffer_samples, numpy_func_output
                )
                self.gradient_update(*batch)

    def at_step_end(self):
        avg_weights = []
        for w1, w2 in zip(
            self.model.trainable_variables, self.avg_model.trainable_variables
        ):
            avg_weight = self.ema_decay * w2 + (1 - self.ema_decay) * w1
            avg_weights.append(avg_weight)
        self.avg_model.set_weights(avg_weights)


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
