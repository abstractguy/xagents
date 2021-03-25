import numpy as np
import tensorflow as tf

from ppo import PPO


class TRPO(PPO):
    def __init__(self, envs, model, n_steps=512, lam=1, *args, **kwargs):
        super(TRPO, self).__init__(envs, model, n_steps, lam, *args, **kwargs)

    @tf.function
    def train_step(self):
        states, actions, returns, values, log_probs = tf.numpy_function(
            self.get_batch, [], 5 * [tf.float32]
        )


if __name__ == '__main__':
    from models import CNNA2C
    from utils import create_gym_env

    envi = create_gym_env('PongNoFrameskip-v4', 16)
    mod = CNNA2C(
        envi[0].observation_space.shape,
        envi[0].action_space.n,
    )
    agn = TRPO(envi, mod)
    agn.train_step()
