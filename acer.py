import numpy as np
import tensorflow as tf

from a2c import A2C
from models import CNNA2C
from utils import ReplayBuffer, create_gym_env


class ACER(A2C):
    def __init__(
        self,
        envs,
        models,
        ema_alpha=0.99,
        buffer_max_size=5000,
        buffer_initial_size=500,
        n_steps=20,
        grad_norm=10,
        replay_ratio=4,
        epsilon=1e-6,
        importance_c=10.0,
        delta=1,
        trust_region=True,
        *args,
        **kwargs,
    ):
        super(ACER, self).__init__(
            envs, models[0], n_steps=n_steps, grad_norm=grad_norm, *args, **kwargs
        )
        self.avg_model = models[1]
        self.ema = tf.train.ExponentialMovingAverage(ema_alpha)
        self.buffers = [
            ReplayBuffer(
                buffer_max_size // self.n_envs,
                buffer_initial_size // self.n_envs,
                batch_size=1,
            )
            for _ in range(self.n_envs)
        ]
        self.batch_indices = tf.range(self.n_steps * self.n_envs, dtype=tf.int64)[
            :, tf.newaxis
        ]
        self.replay_ratio = replay_ratio
        self.epsilon = epsilon
        self.importance_c = importance_c
        self.delta = delta
        self.trust_region = trust_region
        self.tf_batch_dtypes = [tf.uint8] + [tf.float32 for _ in range(4)]
        self.np_batch_dtypes = [np.uint8] + [np.float32 for _ in range(4)]
        self.batch_shapes = [
            (self.n_envs * (self.n_steps + 1), *self.input_shape),
            (self.n_envs * self.n_steps),
            (self.n_envs * self.n_steps),
            (self.n_envs * self.n_steps),
            (self.n_envs * self.n_steps, self.n_actions),
        ]

    def flat_to_steps(self, t, steps=None):
        t = tf.reshape(t, (self.n_envs, steps or self.n_steps, *t.shape[1:]))
        return [
            tf.squeeze(step_t, 1) for step_t in tf.split(t, steps or self.n_steps, 1)
        ]

    def clip_last_step(self, t):
        ts = self.flat_to_steps(t, self.n_steps + 1)
        return tf.reshape(tf.stack(ts[:-1], 1), (-1, *ts[0].shape[1:]))

    @staticmethod
    def gradient_add(g1, g2):
        assert g1 is not None or g2 is not None
        if g1 is not None and g2 is not None:
            return g1 + g2
        if g1 is not None:
            return g1
        return g2

    def calculate_returns(
        self, rewards, dones, selected_critic_logits, values, selected_importance
    ):
        importance_bar = self.flat_to_steps(tf.minimum(1.0, selected_importance))
        dones = self.flat_to_steps(dones)
        rewards = self.flat_to_steps(rewards)
        selected_critic_logits = self.flat_to_steps(selected_critic_logits)
        values = self.flat_to_steps(values, self.n_steps + 1)
        current_return = values[-1]
        returns = []
        for i in reversed(range(self.n_steps)):
            current_return = rewards[i] + self.gamma * current_return * (1.0 - dones[i])
            returns.append(current_return)
            current_return = (
                importance_bar[i] * (current_return - selected_critic_logits[i])
            ) + values[i]
        return tf.reshape(tf.stack(returns[::-1], 1), [-1])

    def update_avg_weights(self):
        avg_variables = [
            self.ema.average(weight).numpy()
            for weight in self.model.trainable_variables
        ]
        self.avg_model.set_weights(avg_variables)

    def store_batch(self, batch):
        for i in range(self.n_envs):
            env_outputs = []
            for item in batch:
                env_outputs.append(item[i])
            self.buffers[i].append(env_outputs)

    def get_batch(self):
        (
            states,
            rewards,
            actions,
            _,
            dones,
            *_,
            actor_logits,
        ) = super(ACER, self).get_batch()
        states.append(self.get_states())
        batch = [states, rewards, actions, dones[1:], actor_logits]
        batch = [
            np.asarray(item, dtype).swapaxes(0, 1)
            for (item, dtype) in zip(batch, self.np_batch_dtypes)
        ]
        self.store_batch(batch)
        return [item.reshape(-1, *item.shape[2:]) for item in batch]

    def calculate_losses(
        self,
        action_probs,
        values,
        returns,
        selected_probs,
        selected_importance,
        selected_critic_logits,
    ):
        entropy = tf.reduce_mean(
            -tf.reduce_sum(
                action_probs * tf.math.log(action_probs + self.epsilon), axis=1
            )
        )
        values = self.clip_last_step(values)
        advantages = returns - values
        log_probs = tf.math.log(selected_probs + self.epsilon)
        action_gain = log_probs * tf.stop_gradient(
            advantages * tf.minimum(self.importance_c, selected_importance)
        )
        action_loss = -tf.reduce_mean(action_gain)
        value_loss = (
            tf.reduce_mean(
                tf.square(tf.stop_gradient(returns) - selected_critic_logits) * 0.5
            )
            * self.value_loss_coef
        )
        if self.trust_region:
            return (
                -(action_loss - self.entropy_coef * entropy)
                * self.n_steps
                * self.n_envs
            ), value_loss
        return (
            action_loss
            + self.value_loss_coef * value_loss
            - self.entropy_coef * entropy
        )

    def calculate_grads(self, tape, losses, action_probs, avg_action_probs):
        if self.trust_region:
            loss, value_loss = losses
            g = tape.gradient(
                loss,
                action_probs,
            )
            k = -avg_action_probs / (action_probs + self.epsilon)
            adj = tf.maximum(
                0.0,
                (tf.reduce_sum(k * g, axis=-1) - self.delta)
                / (tf.reduce_sum(tf.square(k), axis=-1) + self.epsilon),
            )
            g = g - tf.reshape(adj, [self.n_envs * self.n_steps, 1]) * k
            output_grads = -g / (self.n_envs * self.n_steps)
            action_grads = tape.gradient(
                action_probs, self.model.trainable_variables, output_grads
            )
            value_grads = tape.gradient(value_loss, self.model.trainable_variables)
            return [
                self.gradient_add(g1, g2) for (g1, g2) in zip(action_grads, value_grads)
            ]
        return tape.gradient(losses, self.model.trainable_variables)

    def update_gradients(
        self,
        states,
        rewards,
        actions,
        dones,
        previous_action_probs,
    ):
        action_indices = tf.concat(
            (self.batch_indices, tf.cast(actions[:, tf.newaxis], tf.int64)), -1
        )
        with tf.GradientTape(True) as tape:
            *_, critic_logits, _, action_probs = self.model(states)
            *_, avg_action_probs = self.avg_model(states)
            values = tf.reduce_sum(action_probs * critic_logits, axis=-1)
            action_probs = self.clip_last_step(action_probs)
            avg_action_probs = self.clip_last_step(avg_action_probs)
            critic_logits = self.clip_last_step(critic_logits)
            selected_probs = tf.gather_nd(action_probs, action_indices)
            selected_critic_logits = tf.gather_nd(critic_logits, action_indices)
            importance_weights = action_probs / (previous_action_probs + self.epsilon)
            selected_importance = tf.gather_nd(importance_weights, action_indices)
            returns = self.calculate_returns(
                rewards, dones, selected_critic_logits, values, selected_importance
            )
            losses = self.calculate_losses(
                action_probs,
                values,
                returns,
                selected_probs,
                selected_importance,
                selected_critic_logits,
            )
        grads = self.calculate_grads(tape, losses, action_probs, avg_action_probs)
        if self.grad_norm is not None:
            grads, norm_grads = tf.clip_by_global_norm(grads, self.grad_norm)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.ema.apply(self.model.trainable_variables)
        tf.numpy_function(self.update_avg_weights, [], [])

    @tf.function
    def train_step(self):
        batch = tf.numpy_function(self.get_batch, [], self.tf_batch_dtypes)
        for item, shape in zip(batch, self.batch_shapes):
            item.set_shape(shape)
        self.update_gradients(*batch)
        if (
            self.replay_ratio > 0
            and len(self.buffers[0]) >= self.buffers[0].initial_size
        ):
            for _ in range(np.random.poisson(self.replay_ratio)):
                batch = tf.numpy_function(
                    self.concat_buffer_samples,
                    self.np_batch_dtypes,
                    self.tf_batch_dtypes,
                )
                self.update_gradients(*batch)


if __name__ == '__main__':
    sd = 555
    es = create_gym_env('PongNoFrameskip-v4', 2, scale_frames=False)
    ms = [
        CNNA2C(
            (84, 84, 1),
            6,
            actor_activation='softmax',
            critic_units=6,
            seed=sd,
            scale_inputs=True,
        )
        for _ in range(2)
    ]
    agn = ACER(es, ms, seed=sd, optimizer=tf.keras.optimizers.Adam(7e-4))
    agn.fit(19)
