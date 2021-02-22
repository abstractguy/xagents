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
            ReplayBuffer(
                buffer_max_size, batch_size=self.n_envs * self.n_steps, seed=self.seed
            )
            for _ in range(self.n_envs)
        ]
        self.replay_ratio = replay_ratio
        self.epsilon = epsilon
        self.delta = delta
        self.importance_c = importance_c
        self.batch_indices = tf.range(self.n_steps * self.n_envs, dtype=tf.int64)[
            :, tf.newaxis
        ]

    # @tf.function
    def train_step(self):
        batch = self.get_batch()
        (
            states,
            rewards,
            actions,
            critic_logits,
            dones,
            log_probs,
            entropies,
            action_probs,
        ) = [tf.concat(item, 0) for item in batch]
        log_probs = -log_probs
        values = tf.reduce_sum(action_probs * critic_logits, axis=-1)
        action_indices = self.get_action_indices(self.batch_indices, actions)
        selected_action_probs = tf.gather_nd(action_probs, action_indices)
        selected_critic_logits = tf.gather_nd(critic_logits, action_indices)
        importance_ratio = action_probs / (action_probs + self.epsilon)
        selected_ratios = tf.gather_nd(importance_ratio, action_indices)
        importance_bar = tf.minimum(1.0, selected_ratios)
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
        returns = tf.concat(returns, 0)
        titles = [
            'neg log-probs',
            'values',
            'actions',
            'action probs',
            'selected action probs',
            'critic logits',
            'selected critic logits',
            'importance_ratio',
            'selected_ratios',
            'returns',
        ]
        values = [
            log_probs,
            values,
            actions,
            action_probs,
            selected_action_probs,
            critic_logits,
            selected_critic_logits,
            importance_ratio,
            selected_ratios,
            returns,
        ]
        for title, values in zip(titles, values):
            print(f'{title}: {values.shape}\n{values}')
        exit()


if __name__ == '__main__':
    from tensorflow.keras.optimizers import Adam
    from tensorflow_addons.optimizers import MovingAverage

    from models import CNNA2C
    from utils import create_gym_env

    seed = None
    envi = create_gym_env('PongNoFrameskip-v4', 2)
    m = CNNA2C(
        envi[0].observation_space.shape,
        envi[0].action_space.n,
        critic_units=envi[0].action_space.n,
        seed=seed,
        actor_activation='softmax',
    )
    o = MovingAverage(Adam(7e-4))
    agn = ACER(envi, m, optimizer=o, seed=seed)
    agn.fit(19)
