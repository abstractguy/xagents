import numpy as np
import tensorflow as tf

from a2c import A2C


class PPO(A2C):
    def __init__(
        self,
        envs,
        model,
        n_steps=128,
        lam=0.95,
        ppo_epochs=4,
        mini_batches=4,
        advantage_epsilon=1e-8,
        clip_norm=0.1,
        *args,
        **kwargs,
    ):
        """
        Initialize PPO agent.
        Args:
            envs: A list of gym environments.
            model: tf.keras.models.Model used for training.
            n_steps: n-step transition for example given s1, s2, s3, s4 and n_step = 4,
                transition will be s1 -> s4 (defaults to 1, s1 -> s2)
            lam: GAE-Lambda for advantage estimation
            ppo_epochs: Gradient updates per training step.
            mini_batches: Number of mini batches to use per gradient update.
            advantage_epsilon: Epsilon value added to estimated advantage.
            clip_norm: Clipping value passed to tf.clip_by_value()
            *args: args Passed to BaseAgent
            **kwargs: kwargs Passed to BaseAgent
        """
        super(PPO, self).__init__(envs, model, n_steps=n_steps, *args, **kwargs)
        self.lam = lam
        self.ppo_epochs = ppo_epochs
        self.mini_batches = mini_batches
        self.advantage_epsilon = advantage_epsilon
        self.clip_norm = clip_norm
        self.batch_size = self.n_envs * self.n_steps
        self.mini_batch_size = self.batch_size // self.mini_batches

    def calculate_returns(self, states, rewards, values, dones):
        next_values = self.model(states[-1])[-1].numpy()
        advantages = np.zeros_like(rewards)
        last_lam = 0
        values = np.concatenate([values, np.expand_dims(next_values, 0)])
        dones = np.concatenate([dones, np.expand_dims(dones[-1], 0)])
        for step in reversed(range(self.n_steps)):
            next_non_terminal = 1 - dones[step + 1]
            next_values = values[step + 1]
            delta = (
                rewards[step]
                + self.gamma * next_values * next_non_terminal
                - values[step]
            )
            advantages[step] = last_lam = (
                delta + self.gamma * self.lam * next_non_terminal * last_lam
            )
        return advantages + values[:-1]

    def update_step(
        self, states, actions, old_values, returns, old_log_probs, advantages
    ):
        with tf.GradientTape() as tape:
            _, log_probs, entropy, values = self.model(states, actions=actions)
            entropy = tf.reduce_mean(entropy)
            clipped_values = old_values + tf.clip_by_value(
                values - old_values, -self.clip_norm, self.clip_norm
            )
            vf_loss1 = tf.square(values - returns)
            vf_loss2 = tf.square(clipped_values - returns)
            vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_loss1, vf_loss2))
            ratio = tf.exp(log_probs - old_log_probs)
            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * tf.clip_by_value(
                ratio, 1 - self.clip_norm, 1 + self.clip_norm
            )
            pg_loss = tf.reduce_mean(tf.maximum(pg_loss1, pg_loss2))
            loss = (
                pg_loss - entropy * self.entropy_coef + vf_loss * self.value_loss_coef
            )
        grads = tape.gradient(loss, self.model.trainable_variables)
        if self.grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.grad_norm)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def run_ppo_epochs(self, states, actions, returns, old_values, old_log_probs):
        indices = np.arange(self.batch_size)
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for i in range(0, self.batch_size, self.mini_batch_size):
                batch_indices = indices[i : i + self.mini_batch_size]
                mini_batch = [
                    tf.constant(item[batch_indices])
                    for item in [
                        states,
                        actions,
                        returns,
                        old_values,
                        old_log_probs,
                    ]
                ]
                (
                    states_mb,
                    actions_mb,
                    returns_mb,
                    old_values_mb,
                    old_log_probs_mb,
                ) = mini_batch
                advantages_mb = returns_mb - old_values_mb
                (advantages_mb - tf.reduce_mean(advantages_mb)) / (
                    tf.keras.backend.std(advantages_mb) + self.advantage_epsilon
                )
                self.update_step(
                    states_mb,
                    actions_mb,
                    old_values_mb,
                    returns_mb,
                    old_log_probs_mb,
                    advantages_mb,
                )

    def step_transitions(self):
        (
            states,
            rewards,
            actions,
            values,
            dones,
            log_probs,
            _,
        ) = [np.asarray(item, np.float32) for item in self.get_batch()]
        returns = self.calculate_returns(states, rewards, values, dones)
        ppo_batch = [
            a.swapaxes(0, 1).reshape(a.shape[0] * a.shape[1], *a.shape[2:])
            for a in [states, actions, returns, values, log_probs]
        ]
        self.run_ppo_epochs(*ppo_batch)

    @tf.function
    def train_step(self):
        tf.numpy_function(self.step_transitions, [], [])


if __name__ == '__main__':
    from tensorflow.keras.optimizers import Adam

    from models import CNNA2C
    from utils import create_gym_env

    envi = create_gym_env('PongNoFrameskip-v4', 16)
    mod = CNNA2C(envi[0].observation_space.shape, envi[0].action_space.n)
    agn = PPO(envi, mod, optimizer=Adam(25e-5))
    agn.fit(19)
