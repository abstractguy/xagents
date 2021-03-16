import numpy as np
import tensorflow as tf

from a2c import A2C
from models import CNNA2C
from utils import ReplayBuffer, create_gym_env


class ACER(A2C):
    def __init__(
        self, env, models, n_steps=20, seed=None, buffer_size=50000, ema_alpha=0.99
    ):
        super(ACER, self).__init__(env, models[0], n_steps=n_steps, seed=seed)
        self.avg_model = models[1]
        self.buffers = [
            ReplayBuffer(buffer_size // len(env), 500, batch_size=1, seed=seed)
            for _ in range(len(env))
        ]
        self.ema = tf.train.ExponentialMovingAverage(ema_alpha)
        self.batch_indices = tf.range(self.n_steps * self.n_envs, dtype=tf.int64)[
            :, tf.newaxis
        ]

    @staticmethod
    def gradient_add(g1, g2):
        assert not (g1 is None and g2 is None)
        if g1 is None:
            return g2
        elif g2 is None:
            return g1
        else:
            return g1 + g2

    def q_retrace(self, rewards, dones, q_i, values, rho_i):
        rho_bar = tf.unstack(
            tf.reshape(tf.minimum(1.0, rho_i), (self.n_envs, self.n_steps)), axis=1
        )
        dones = tf.unstack(tf.reshape(dones, (self.n_envs, self.n_steps)), axis=1)
        rewards = tf.unstack(tf.reshape(rewards, (self.n_envs, self.n_steps)), axis=1)
        q_i = tf.unstack(tf.reshape(q_i, (self.n_envs, self.n_steps)), axis=1)
        values = tf.unstack(tf.reshape(values, (self.n_envs, self.n_steps + 1)), axis=1)
        qret = values[-1]
        qrets = []
        for i in range(self.n_steps - 1, -1, -1):
            qret = rewards[i] + self.gamma * qret * (1.0 - dones[i])
            qrets.append(qret)
            qret = (rho_bar[i] * (qret - q_i[i])) + values[i]
        return tf.reshape(tf.stack(qrets[::-1], 1), [-1])

    def run(
        self,
    ):
        (
            states,
            rewards,
            actions,
            _,
            dones,
            *_,
            actor_logits,
        ) = self.get_batch()
        states.append(self.get_states())
        states = np.asarray(states, np.uint8).swapaxes(1, 0)
        actions = np.asarray(actions, np.int32).swapaxes(1, 0)
        rewards = np.asarray(rewards, dtype=np.float32).swapaxes(1, 0)
        actor_logits = np.asarray(actor_logits, dtype=np.float32).swapaxes(1, 0)
        dones = np.asarray(dones[1:], dtype=np.float32).swapaxes(1, 0)
        return states, actions, rewards, actor_logits, dones

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

    def call(self, on_policy):
        if on_policy:
            batch = self.run()
            if self.buffers is not None:
                for i in range(self.n_envs):
                    env_outputs = []
                    for item in batch:
                        env_outputs.append(item[i])
                    self.buffers[i].append(env_outputs)
            obs, actions, rewards, mus, dones = [
                item.reshape(-1, *item.shape[2:]) for item in batch
            ]
        else:
            obs, actions, rewards, mus, dones = self.concat_buffer_samples()
        self.train(obs, actions, mus, rewards, dones)
        self.update_avg_weights()

    @tf.function
    def train(
        self,
        obs,
        actions,
        mus,
        rewards,
        dones,
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
            f = tf.squeeze(train_model_p[: -self.n_envs])
            f_pol = tf.squeeze(polyak_model_p[: -self.n_envs])
            q = tf.squeeze(train_q[: -self.n_envs])
            f_i = tf.gather_nd(f, action_indices)
            q_i = tf.gather_nd(q, action_indices)
            rho = f / (mus + eps)
            rho_i = tf.gather_nd(rho, action_indices)
            qret = self.q_retrace(rewards, dones, q_i, v, rho_i)
            entropy = tf.reduce_mean(-tf.reduce_sum(f * tf.math.log(f + eps), axis=1))
            v = v[: -self.n_envs]
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


def learn(
    env,
    seed=None,
    total_timesteps=int(80e6),
    replay_ratio=4,
):
    ms = [
        CNNA2C(
            (84, 84, 1),
            6,
            actor_activation='softmax',
            critic_units=6,
            seed=seed,
            scale_inputs=True,
        )
        for _ in range(2)
    ]
    agn = ACER(env, ms, seed=seed)
    agn.init_training(19, *[None] * 3)
    for _ in range(0, total_timesteps):
        agn.call(True)
        agn.check_episodes()
        if replay_ratio > 0 and len(agn.buffers[0]) >= agn.buffers[0].initial_size:
            n = np.random.poisson(replay_ratio)
            for _ in range(n):
                agn.call(False)


if __name__ == '__main__':
    seeed = None
    envs = create_gym_env('PongNoFrameskip-v4', 2, scale_frames=False)
    learn(envs, seed=seeed)
