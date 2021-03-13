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

    # @tf.function
    def train_step(self):
        batch = self.get_batch()

    def update_avg_weights(self):
        avg_variables = [
            self.ema.average(weight).numpy()
            for weight in self.model.trainable_variables
        ]
        self.avg_model.set_weights(avg_variables)

    def at_step_end(self):
        self.update_avg_weights()


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
