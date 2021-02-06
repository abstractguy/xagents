import numpy as np
import tensorflow as tf

from base_agent import BaseAgent


class PPO(BaseAgent):
    def __init__(
        self,
        envs,
        model,
        transition_steps=128,
        lam=0.95,
        ppo_epochs=4,
        mini_batches=4,
        advantage_epsilon=1e-5,
        clip_norm=0.1,
        entropy_coef=0.01,
        vf_coef=0.5,
        grad_norm=0.5,
        *args,
        **kwargs,
    ):
        super(PPO, self).__init__(
            envs, model, transition_steps=transition_steps, *args, **kwargs
        )
        self.lam = lam
        self.ppo_epochs = ppo_epochs
        self.mini_batches = mini_batches
        self.advantage_epsilon = advantage_epsilon
        self.clip_norm = clip_norm
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.grad_norm = grad_norm
        self.batch_size = self.n_envs * self.transition_steps
        self.mini_batch_size = self.batch_size // self.mini_batches

    def get_states(self):
        """
        Get self.states
        Returns:
            self.states as numpy array
        """
        return np.array(self.states, np.float32)

    def get_dones(self):
        return np.array(self.dones, np.float32)

    @staticmethod
    def sf01(arr):
        s = arr.shape
        return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

    def get_batch(self):
        batch = states, rewards, actions, values, dones, log_probs = [
            [] for _ in range(6)
        ]
        step_states = tf.numpy_function(self.get_states, [], tf.float32)
        step_dones = tf.numpy_function(self.get_dones, [], tf.float32)
        for _ in range(self.transition_steps):
            step_actions, step_log_probs, _, step_values = self.model(step_states)
            states.append(step_states)
            actions.append(step_actions)
            values.append(step_values)
            log_probs.append(step_log_probs)
            dones.append(step_dones)
            step_states, step_rewards, step_dones = tf.numpy_function(
                self.step_envs, [step_actions], [tf.float32 for _ in range(3)]
            )
            rewards.append(step_rewards)
        return [np.asarray(item, np.float32) for item in batch]

    def calculate_returns(self, batch):
        states, rewards, actions, values, dones, log_probs = batch
        next_values = self.model(states[-1])[-1].numpy()
        advantages = np.zeros_like(rewards)
        last_lam = 0
        for step in reversed(range(self.transition_steps)):
            if step == self.transition_steps - 1:
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
                    # kl = 0.5 * tf.reduce_mean(tf.square(log_probs - old_log_probs))
                    # clip_frac = tf.reduce_mean(
                    #     tf.cast(
                    #         tf.greater(tf.abs(ratio - 1.0), self.clip_norm), tf.float32
                    #     )
                    # )
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
        batch = states, rewards, actions, values, dones, log_probs = self.get_batch()
        returns = self.calculate_returns(batch)
        ppo_batch = [
            self.sf01(item)
            for item in [states, actions, returns, dones, values, log_probs]
        ]
        self.run_ppo_epochs(ppo_batch)
        return True

    @tf.function
    def train_step(self):
        tf.numpy_function(self.step_transitions, [], tf.bool)

    def fit(
        self,
        target_reward,
        time_steps,
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

    envi = create_gym_env('PongNoFrameskip-v4', 2)
    mod = CNNA2C(envi[0].observation_space.shape, envi[0].action_space.n)
    agn = PPO(envi, mod, optimizer=Adam(25e-5))
    agn.fit(19, 2e7)
