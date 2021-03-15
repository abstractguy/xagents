import time
from collections import deque

import numpy as np
import tensorflow as tf

from models import CNNA2C
from utils import ReplayBuffer, create_gym_env
from tensorflow.keras.optimizers import Adam


class Runner:
    def __init__(self, env, model, n_steps):
        self.env = env
        self.model = model
        self.n_envs = n_envs = len(env)
        self.batch_ob_shape = (n_envs * n_steps,) + env[0].observation_space.shape
        self.obs = np.asarray([e.reset() for e in env])
        self.n_steps = n_steps
        self.dones = [False for _ in range(n_envs)]
        self.avg_model = model
        self.n_actions = env[0].action_space.n
        n_envs = self.n_envs
        self.n_batches = n_envs * n_steps
        self.batch_ob_shape = (n_envs * (n_steps + 1),) + env[0].observation_space.shape
        self.ac_dtype = env[0].action_space.dtype
        self.nc = 1

    def run(
        self,
        episode_rewards,
        total_rewards,
    ):
        mb_obs, mb_actions, mb_mus, mb_dones, mb_rewards = [], [], [], [], []
        for _ in range(self.n_steps):
            actions, *_, mus = self.model(self.obs)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_mus.append(mus)
            mb_dones.append(self.dones)
            obs, rewards, dones, = (
                [],
                [],
                [],
            )
            for i, (e, a) in enumerate(zip(self.env, actions)):
                s, r, d, _ = e.step(a)
                if d:
                    obs.append(e.reset())
                    total_rewards.append(episode_rewards[i])
                    episode_rewards[i] = 0

                else:
                    obs.append(s)
                episode_rewards[i] += r
                rewards.append(r)
                dones.append(d)
            obs, rewards, dones = [np.asarray(item) for item in [obs, rewards, dones]]
            self.dones = dones
            self.obs = obs
            mb_rewards.append(rewards)
        mb_obs.append(np.copy(self.obs))
        mb_dones.append(self.dones)
        mb_obs = np.asarray(mb_obs, np.uint8).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=self.ac_dtype).swapaxes(1, 0)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_mus = np.asarray(mb_mus, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.float32).swapaxes(1, 0)
        mb_dones = mb_dones[:, 1:]
        return mb_obs, mb_actions, mb_rewards, mb_mus, mb_dones


def cat_entropy_softmax(p0):
    return -tf.reduce_sum(input_tensor=p0 * tf.math.log(p0 + 1e-6), axis=1)


def get_by_index(x, idx):
    assert len(x.shape) == 2
    assert len(idx.shape) == 1
    idx_flattened = tf.range(0, x.shape[0]) * x.shape[1] + tf.cast(idx, tf.int32)
    y = tf.gather(tf.reshape(x, [-1]), idx_flattened)
    return y


def gradient_add(g1, g2):
    assert not (g1 is None and g2 is None)
    if g1 is None:
        return g2
    elif g2 is None:
        return g1
    else:
        return g1 + g2
    pass


def q_retrace(rewards, dones, q_i, values, rho_i, n_envs, n_steps, gamma):
    rho_bar = tf.unstack(tf.reshape(tf.minimum(1.0, rho_i), (n_envs, n_steps)), axis=1)
    dones = tf.unstack(tf.reshape(dones, (n_envs, n_steps)), axis=1)
    rewards = tf.unstack(tf.reshape(rewards, (n_envs, n_steps)), axis=1)
    q_i = tf.unstack(tf.reshape(q_i, (n_envs, n_steps)), axis=1)
    values = tf.unstack(tf.reshape(values, (n_envs, n_steps + 1)), axis=1)
    qret = values[-1]
    qrets = []
    for i in range(n_steps - 1, -1, -1):
        qret = rewards[i] + gamma * qret * (1.0 - dones[i])
        qrets.append(qret)
        qret = (rho_bar[i] * (qret - q_i[i])) + values[i]
    return tf.reshape(tf.stack(qrets[::-1], 1), [-1])


class Acer:
    def __init__(
        self,
        env,
        runner,
        v2_models,
        buffers,
        log_interval,
        reward_buffer_size=100,
        metric_digits=2,
        n_steps=20,
        ema_alpha=0.99,
    ):
        self.n_actions = env[0].action_space.n
        self.input_shape = env[0].observation_space.shape
        self.n_envs = len(env)
        self.n_steps = n_steps
        self.runner = runner
        self.model, self.avg_model = v2_models
        self.buffers = buffers
        self.log_interval = log_interval
        self.tstart = None
        self.steps = None
        self.total_rewards = deque(maxlen=reward_buffer_size)
        self.metric_digits = metric_digits
        self.best_reward = -float('inf')
        self.mean_reward = -float('inf')
        self.episode_rewards = np.zeros(runner.n_envs)
        self.ema = tf.train.ExponentialMovingAverage(ema_alpha)
        self.training_return_names = [
            'loss',
            'loss_q',
            'entropy',
            'loss_f',
        ]

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
            obs, actions, rewards, mus, dones = self.runner.run(
                self.episode_rewards, self.total_rewards
            )
            if self.buffers is not None:
                for i in range(self.n_envs):
                    env_outputs = []
                    for item in [obs, actions, rewards, mus, dones]:
                        env_outputs.append(item[i])
                    self.buffers[i].append(env_outputs)
            obs = obs.reshape(self.runner.batch_ob_shape)
            actions = actions.reshape([self.runner.n_batches])
            rewards = rewards.reshape([self.runner.n_batches])
            mus = mus.reshape([self.runner.n_batches, self.runner.n_actions])
            dones = dones.reshape([self.runner.n_batches])
        else:
            obs, actions, rewards, mus, dones = self.concat_buffer_samples()
        values_ops = self.train(obs, actions, mus, rewards, dones)
        self.update_avg_weights()
        if on_policy and (
            int(self.steps / self.runner.n_batches) % self.log_interval == 0
        ):
            print("total_timesteps", self.steps)
            print("fps", int(self.steps / (time.time() - self.tstart)))
            print(
                "mean_episode_reward",
                np.around(np.mean(self.total_rewards or [0]), self.metric_digits),
            )
            for name, val in zip(self.training_return_names, values_ops):
                print(f'{name}: {np.around(float(val), 2)}')
            print(30 * '=')

    @tf.function
    def train(
        self,
        obs,
        actions,
        mus,
        rewards,
        dones,
        eps=1e-6,
        gamma=0.99,
        c=10.0,
        q_coef=0.5,
        ent_coef=0.01,
        delta=1,
        trust_region=True,
        max_grad_norm=10,
    ):
        with tf.GradientTape(persistent=True) as tape:
            *_, train_q, _, train_model_p = self.model(obs)
            *_, polyak_q, _, polyak_model_p = self.avg_model(obs)
            v = tf.reduce_sum(input_tensor=train_model_p * train_q, axis=-1)
            f = tf.squeeze(train_model_p[: -self.n_envs])
            f_pol = tf.squeeze(polyak_model_p[: -self.n_envs])
            q = tf.squeeze(train_q[: -self.n_envs])
            f_i = get_by_index(f, actions)
            q_i = get_by_index(q, actions)
            rho = f / (mus + eps)
            rho_i = get_by_index(rho, actions)
            qret = q_retrace(
                rewards, dones, q_i, v, rho_i, self.n_envs, self.n_steps, gamma
            )
            entropy = tf.reduce_mean(input_tensor=cat_entropy_softmax(f))
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
            grads = [gradient_add(g1, g2) for (g1, g2) in zip(grads_policy, grads_q)]
        else:
            grads = tape.gradient(loss, self.model.trainable_variables)
        if max_grad_norm is not None:
            grads, norm_grads = tf.clip_by_global_norm(grads, max_grad_norm)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.ema.apply(self.model.trainable_variables)
        return [
            loss,
            loss_q,
            entropy,
            loss_f,
        ]


def learn(
    env,
    seed=None,
    n_steps=20,
    total_timesteps=int(80e6),
    log_interval=100,
    buffer_size=10000,
    replay_ratio=4,
):
    n_envs = len(env)
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
    ms[0].compile(optimizer=Adam())
    runner = Runner(env=env, model=ms[0], n_steps=n_steps)
    buffers = [
        ReplayBuffer(buffer_size // len(env), 500, batch_size=1, seed=seed)
        for _ in range(len(env))
    ]
    n_batches = n_envs * n_steps
    acer = Acer(env, runner, ms, buffers, log_interval)
    acer.tstart = time.time()
    for acer.steps in range(0, total_timesteps, n_batches):
        acer.call(True)
        if replay_ratio > 0 and len(buffers[0]) >= buffers[0].initial_size:
            n = np.random.poisson(replay_ratio)
            for _ in range(n):
                acer.call(False)


if __name__ == '__main__':
    envs = create_gym_env('PongNoFrameskip-v4', 2, scale_frames=False)
    learn(envs)
