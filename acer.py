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
        n_steps=20,
        grad_norm=10,
        buffer_max_size=5000,
        buffer_initial_size=500,
        replay_ratio=4,
        *args,
        **kwargs,
    ):
        super(ACER, self).__init__(
            envs, models[0], n_steps=n_steps, grad_norm=grad_norm, *args, **kwargs
        )
        self.avg_model = models[1]
        self.buffers = [
            ReplayBuffer(
                buffer_max_size // self.n_envs,
                buffer_initial_size // self.n_envs,
                batch_size=1,
            )
            for _ in range(self.n_envs)
        ]
        self.ema = tf.train.ExponentialMovingAverage(ema_alpha)
        self.batch_indices = tf.range(self.n_steps * self.n_envs, dtype=tf.int64)[
            :, tf.newaxis
        ]
        self.replay_ratio = replay_ratio
        self.tf_batch_dtypes = [tf.uint8] + [tf.float32 for _ in range(4)]
        self.np_batch_dtypes = [np.uint8] + [np.float32 for _ in range(4)]
        self.batch_shapes = [
            (self.n_envs * (self.n_steps + 1), *self.input_shape),
            (self.n_envs * self.n_steps),
            (self.n_envs * self.n_steps),
            (self.n_envs * self.n_steps),
            (self.n_envs * self.n_steps, self.available_actions),
        ]

    def flat_to_steps(self, t, steps=None):
        t = tf.reshape(t, (self.n_envs, steps or self.n_steps, *t.shape[1:]))
        return [tf.squeeze(v, 1) for v in tf.split(t, steps or self.n_steps, 1)]

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

    def q_retrace(self, rewards, dones, q_i, values, rho_i):
        rho_bar = self.flat_to_steps(tf.minimum(1.0, rho_i))
        dones = self.flat_to_steps(dones)
        rewards = self.flat_to_steps(rewards)
        q_i = self.flat_to_steps(q_i)
        values = self.flat_to_steps(values, self.n_steps + 1)
        qret = values[-1]
        qrets = []
        for i in range(self.n_steps - 1, -1, -1):
            qret = rewards[i] + self.gamma * qret * (1.0 - dones[i])
            qrets.append(qret)
            qret = (rho_bar[i] * (qret - q_i[i])) + values[i]
        return tf.reshape(tf.stack(qrets[::-1], 1), [-1])

    def update_avg_weights(self):
        avg_variables = [
            self.ema.average(weight).numpy()
            for weight in self.model.trainable_variables
        ]
        self.avg_model.set_weights(avg_variables)

    def concat_buffer_samples(self):
        batches = []
        for i in range(self.n_envs):
            batch = self.buffers[i].get_sample()
            batches.append(batch)
        if len(batches) > 1:
            return [np.concatenate(item) for item in zip(*batches)]
        return batches[0]

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

    def train(
        self,
        obs,
        rewards,
        actions,
        dones,
        mus,
        eps=1e-6,
        c=10.0,
        q_coef=0.5,
        ent_coef=0.01,
        delta=1,
        trust_region=True,
        max_grad_norm=10,
    ):
        action_indices = tf.concat(
            (self.batch_indices, tf.cast(actions[:, tf.newaxis], tf.int64)), -1
        )
        with tf.GradientTape(persistent=True) as tape:
            *_, train_q, _, train_model_p = self.model(obs)
            *_, polyak_q, _, polyak_model_p = self.avg_model(obs)
            v = tf.reduce_sum(input_tensor=train_model_p * train_q, axis=-1)
            f = self.clip_last_step(train_model_p)
            f_pol = self.clip_last_step(polyak_model_p)
            q = self.clip_last_step(train_q)
            f_i = tf.gather_nd(f, action_indices)
            q_i = tf.gather_nd(q, action_indices)
            rho = f / (mus + eps)
            rho_i = tf.gather_nd(rho, action_indices)
            qret = self.q_retrace(rewards, dones, q_i, v, rho_i)
            entropy = tf.reduce_mean(-tf.reduce_sum(f * tf.math.log(f + eps), axis=1))
            v = self.clip_last_step(v)
            adv = qret - v
            logf = tf.math.log(f_i + eps)
            gain_f = logf * tf.stop_gradient(adv * tf.minimum(c, rho_i))
            loss_f = -tf.reduce_mean(input_tensor=gain_f)
            loss_q = (
                tf.reduce_mean(
                    input_tensor=tf.square(tf.stop_gradient(qret) - q_i) * 0.5
                )
                * q_coef
            )
            if trust_region:
                loss = -(loss_f - ent_coef * entropy) * self.n_steps * self.n_envs
            else:
                loss = loss_f + q_coef * loss_q - ent_coef * entropy
        if trust_region:
            g = tape.gradient(
                loss,
                f,
            )
            k = -f_pol / (f + eps)
            adj = tf.maximum(
                0.0,
                (tf.reduce_sum(input_tensor=k * g, axis=-1) - delta)
                / (tf.reduce_sum(input_tensor=tf.square(k), axis=-1) + eps),
            )
            g = g - tf.reshape(adj, [self.n_envs * self.n_steps, 1]) * k
            grads_f = -g / (self.n_envs * self.n_steps)
            grads_policy = tape.gradient(f, self.model.trainable_variables, grads_f)
            grads_q = tape.gradient(loss_q, self.model.trainable_variables)
            grads = [
                self.gradient_add(g1, g2) for (g1, g2) in zip(grads_policy, grads_q)
            ]
        else:
            grads = tape.gradient(loss, self.model.trainable_variables)
        if max_grad_norm is not None:
            grads, norm_grads = tf.clip_by_global_norm(grads, max_grad_norm)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.ema.apply(self.model.trainable_variables)
        tf.numpy_function(self.update_avg_weights, [], [])

    @tf.function
    def train_step(self):
        batch = tf.numpy_function(self.get_batch, [], self.tf_batch_dtypes)
        for item, shape in zip(batch, self.batch_shapes):
            item.set_shape(shape)
        self.train(*batch)
        if (
            self.replay_ratio > 0
            and len(self.buffers[0]) >= self.buffers[0].initial_size
        ):
            for _ in range(np.random.poisson(self.replay_ratio)):
                batch = tf.numpy_function(
                    self.concat_buffer_samples, [], self.tf_batch_dtypes
                )
                self.train(*batch)


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
    agn = ACER(es, ms, seed=sd)
    agn.fit(19)
