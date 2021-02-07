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
        advantage_epsilon=1e-5,
        clip_norm=0.1,
        *args,
        **kwargs,
    ):
        super(PPO, self).__init__(envs, model, n_steps=n_steps, *args, **kwargs)
        assert self.model.neg_log_probs, 'PPO model neg_log_probs should be set to True'
        self.lam = lam
        self.ppo_epochs = ppo_epochs
        self.mini_batches = mini_batches
        self.advantage_epsilon = advantage_epsilon
        self.clip_norm = clip_norm
        self.batch_size = self.n_envs * self.n_steps
        self.mini_batch_size = self.batch_size // self.mini_batches

    @staticmethod
    def squeeze_transition_item(a):
        shape = a.shape
        return a.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def calculate_returns(self, batch):
        states, rewards, actions, values, dones, log_probs = batch
        next_values = self.model(states[-1])[-1].numpy()
        advantages = np.zeros_like(rewards)
        last_lam = 0
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                next_non_terminal = 1 - dones[-1]
            else:
                next_non_terminal = 1.0 - dones[step + 1]
                next_values = values[step + 1]
            delta = (
                rewards[step]
                + self.gamma * next_values * next_non_terminal
                - values[step]
            )
            advantages[step] = last_lam = (
                delta + self.gamma * self.lam * next_non_terminal * last_lam
            )
        return advantages + values

    def run_ppo_epochs(self, batch):
        indices = np.arange(self.batch_size)
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for i in range(0, self.batch_size, self.mini_batch_size):
                batch_indices = indices[i : i + self.mini_batch_size]
                mini_batch = [tf.constant(item[batch_indices]) for item in batch]
                states, actions, returns, masks, old_values, old_log_probs = mini_batch
                advantages = returns - old_values
                (advantages - tf.reduce_mean(advantages)) / (
                    tf.keras.backend.std(advantages) + 1e-8
                )
                with tf.GradientTape() as tape:
                    _, log_probs, entropy, values = self.model(states, actions=actions)
                    entropy = tf.reduce_mean(entropy)
                    clipped_values = old_values + tf.clip_by_value(
                        values - old_values, -self.clip_norm, self.clip_norm
                    )
                    vf_loss1 = tf.square(values - returns)
                    vf_loss2 = tf.square(clipped_values - returns)
                    vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_loss1, vf_loss2))
                    ratio = tf.exp(old_log_probs - log_probs)
                    pg_loss1 = -advantages * ratio
                    pg_loss2 = -advantages * tf.clip_by_value(
                        ratio, 1 - self.clip_norm, 1 + self.clip_norm
                    )
                    pg_loss = tf.reduce_mean(tf.maximum(pg_loss1, pg_loss2))
                    loss = (
                        pg_loss - entropy * self.entropy_coef + vf_loss * self.vf_coef
                    )
                grads = tape.gradient(loss, self.model.trainable_variables)
                if self.grad_norm is not None:
                    grads, _ = tf.clip_by_global_norm(grads, self.grad_norm)
                self.model.optimizer.apply_gradients(
                    zip(grads, self.model.trainable_variables)
                )

    def step_transitions(self):
        batch = (
            states,
            rewards,
            actions,
            values,
            dones,
            log_probs,
            entropies,
        ) = [np.asarray(item, np.float32) for item in self.get_batch()]
        returns = self.calculate_returns(batch[:-1])
        ppo_batch = [
            self.squeeze_transition_item(item)
            for item in [states, actions, returns, dones, values, log_probs]
        ]
        self.run_ppo_epochs(ppo_batch)

    @tf.function
    def train_step(self):
        tf.numpy_function(self.step_transitions, [], [])

    def fit(
        self,
        target_reward,
        time_steps=2e7,
        max_steps=None,
        monitor_session=None,
        weights=None,
    ):
        self.init_training(target_reward, max_steps, monitor_session, weights, None)
        updates = 0
        n_batches = self.batch_size
        n_updates = time_steps // n_batches
        while True:
            updates += 1
            frac = 1.0 - (updates - 1.0) / n_updates
            self.model.optimizer.learning_rate.assign(
                self.model.optimizer.learning_rate * frac
            )
            self.check_episodes()
            if self.training_done():
                break
            self.train_step()


if __name__ == '__main__':
    from tensorflow.keras.optimizers import Adam

    from models import CNNA2C
    from utils import create_gym_env

    envi = create_gym_env('PongNoFrameskip-v4', 16)
    mod = CNNA2C(
        envi[0].observation_space.shape, envi[0].action_space.n, neg_log_probs=True
    )
    agn = PPO(envi, mod, optimizer=Adam(25e-5))
    agn.fit(19)
