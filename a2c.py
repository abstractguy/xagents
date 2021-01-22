from collections import deque
from time import perf_counter

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from models import CNNA2C
from utils import create_gym_env


class A2C:
    def __init__(
        self,
        envs,
        entropy_coef,
        value_loss_coef,
        gamma,
        learning_rate,
        transition_steps,
        reward_buffer_size,
    ):
        assert envs, 'No environments given'
        self.envs = envs
        self.n_envs = len(envs)
        self.input_shape = self.envs[0].observation_space.shape
        self.n_actions = self.envs[0].action_space.n
        self.model = CNNA2C(self.input_shape, self.n_actions)
        self.transition_steps = transition_steps
        self.total_rewards = deque(maxlen=reward_buffer_size * self.n_envs)
        self.best_reward = -float('inf')
        self.mean_reward = -float('inf')
        self.states = [
            None
        ] * self.n_envs  # np.zeros((self.n_envs, *self.input_shape), np.float32)
        self.masks = np.ones(self.n_envs)
        self.reset_envs()
        self.steps = 0
        self.games = 0
        self.gamma = gamma
        self.episode_rewards = np.zeros(self.n_envs)
        self.last_reset_step = 0
        self.frame_speed = 0
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.optimizer = tfa.optimizers.RectifiedAdam(
            learning_rate=learning_rate, epsilon=1e-5, beta_1=0.0, beta_2=0.99
        )
        self.done_envs = []

    def reset_envs(self):
        for i, env in enumerate(self.envs):
            self.states[i] = env.reset()

    def get_states(self):
        return np.array(self.states, np.float32)

    def step_envs(self, actions):
        observations = []
        for (i, env), action in zip(enumerate(self.envs), actions):
            state, reward, done, _ = env.step(action)
            self.states[i] = state
            self.steps += 1
            observations.append((state, reward, done))
            self.episode_rewards[i] += reward
            if done:
                self.done_envs.append(1)
                self.total_rewards.append(self.episode_rewards[i])
                self.games += 1
                self.episode_rewards[i] = 0
                self.states[i] = env.reset()

        return [np.array(item, np.float32) for item in zip(*observations)]

    @tf.function
    def update(self):
        masks = []
        rewards = []
        values = []
        log_probs = []
        entropies = []
        obs = tf.numpy_function(func=self.get_states, inp=[], Tout=tf.float32)
        with tf.GradientTape() as tape:
            for j in range(self.transition_steps):
                action, log_prob, entropy, value = self.model(obs)
                obs, reward, done = tf.numpy_function(
                    func=self.step_envs,
                    inp=[action],
                    Tout=(tf.float32, tf.float32, tf.float32),
                )
                mask = 1 - done
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                masks.append(mask)
                entropies.append(entropy)
            next_value = self.model(obs)[-1]
            returns = [next_value]
            for j in reversed(range(self.transition_steps)):
                returns.insert(0, rewards[j] + masks[j] * self.gamma * returns[0])
            value_loss = 0.0
            action_loss = 0.0
            entropy_loss = 0.0
            for j in range(self.transition_steps):
                advantages = tf.stop_gradient(returns[j]) - values[j]
                value_loss += tf.reduce_mean(tf.square(advantages))
                action_loss += -tf.reduce_mean(
                    tf.stop_gradient(advantages) * log_probs[j]
                )
                entropy_loss += tf.reduce_mean(entropies[j])
            value_loss /= self.transition_steps
            action_loss /= self.transition_steps
            entropy_loss /= self.transition_steps
            loss = (
                self.value_loss_coef * value_loss
                + action_loss
                - entropy_loss * self.entropy_coef
            )
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return value_loss, action_loss, entropy_loss

    def display_metrics(self):
        display_titles = (
            'frame',
            'games',
            'speed',
            'mean reward',
            'best reward',
            'episode rewards',
        )
        display_values = (
            self.steps,
            self.games,
            f'{round(self.frame_speed)} steps/s',
            self.mean_reward,
            self.best_reward,
            list(self.total_rewards)[-self.n_envs :],
        )
        display = (
            f'{title}: {value}' for title, value in zip(display_titles, display_values)
        )
        print(', '.join(display))

    def update_metrics(self, start_time):
        self.frame_speed = (self.steps - self.last_reset_step) / (
            perf_counter() - start_time
        )
        self.last_reset_step = self.steps
        self.mean_reward = np.around(np.mean(self.total_rewards), 2)
        self.best_reward = max(self.mean_reward, self.best_reward)

    def fit(self, target_reward):
        start_time = perf_counter()
        while True:
            if self.mean_reward >= target_reward:
                print(f'Reward achieved in {self.steps} steps!')
                break
            if len(self.done_envs) == self.n_envs:
                self.update_metrics(start_time)
                start_time = perf_counter()
                self.display_metrics()
                self.done_envs.clear()
            self.update()


if __name__ == '__main__':
    ens = create_gym_env('PongNoFrameskip-v4', 16)
    ac = A2C(ens, 0.01, 0.5, 0.99, 0.00025, 5, 100)
    ac.fit(18)
