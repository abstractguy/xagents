import numpy as np
import tensorflow as tf
import tqdm
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Model

from utils import create_gym_env


class A2C:
    def __init__(self, env, seed=None, fc_units=512, gamma=0.99):
        self.env = env
        self.available_actions = env.action_space.n
        self.model = self.create_model(fc_units)
        self.env.seed(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)
        self.division_eps = np.finfo(np.float32).eps.item()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.gamma = gamma
        self.steps = 0

    def create_model(self, fc_units):
        x0 = Input(self.env.observation_space.shape)
        x = Conv2D(32, 8, 4, activation='relu')(x0)
        x = Conv2D(64, 4, 2, activation='relu')(x)
        x = Conv2D(32, 3, 1, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(fc_units, activation='relu')(x)
        actor = Dense(self.available_actions)(x)
        critic = Dense(1)(actor)
        model = Model(x0, [actor, critic])
        model.call = tf.function(model.call)
        return model

    def env_step(self, action):
        state, reward, done, _ = self.env.step(action)
        return (
            state.astype(np.float32),
            np.array(reward, np.int32),
            np.array(done, np.int32),
        )

    def tf_env_step(self, action):
        self.steps += 1
        return tf.numpy_function(
            self.env_step, [action], [tf.float32, tf.int32, tf.int32]
        )

    def play_episode(self, initial_state, model, max_steps):
        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        initial_state_shape = initial_state.shape
        state = initial_state
        for t in tf.range(max_steps):
            state = tf.expand_dims(state, 0)
            action_logits_t, value = model(state)
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            action_probs_t = tf.nn.softmax(action_logits_t)
            values = values.write(t, tf.squeeze(value))
            action_probs = action_probs.write(t, action_probs_t[0, action])
            state, reward, done = self.tf_env_step(action)
            state.set_shape(initial_state_shape)
            rewards = rewards.write(t, reward)
            if tf.cast(done, tf.bool):
                break
        return [item.stack() for item in [action_probs, values, rewards]]

    def get_returns(self, rewards, gamma, standardize=True):
        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]
        if standardize:
            returns = (returns - tf.math.reduce_mean(returns)) / (
                tf.math.reduce_std(returns) + self.division_eps
            )
        return returns

    @staticmethod
    def compute_loss(action_probs, values, returns):
        advantage = returns - values
        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)
        critic_loss = Huber(reduction=tf.keras.losses.Reduction.SUM)(values, returns)
        return actor_loss + critic_loss

    @tf.function
    def train_step(
        self,
        initial_state,
        max_steps_per_episode,
    ):
        with tf.GradientTape() as tape:
            action_probs, values, rewards = self.play_episode(
                initial_state, self.model, max_steps_per_episode
            )
            returns = self.get_returns(rewards, self.gamma)
            action_probs, values, returns = [
                tf.expand_dims(x, 1) for x in [action_probs, values, returns]
            ]
            loss = self.compute_loss(action_probs, values, returns)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        episode_reward = tf.math.reduce_sum(rewards)
        return episode_reward

    def fit(self, target_reward, max_episodes=10000, max_steps_per_episode=10000):
        running_reward = 0
        with tqdm.trange(max_episodes) as t:
            for i in t:
                initial_state = tf.constant(self.env.reset(), dtype=tf.float32)
                episode_reward = int(
                    self.train_step(
                        initial_state,
                        max_steps_per_episode,
                    )
                )

                running_reward = episode_reward * 0.01 + running_reward * 0.99

                t.set_description(f'Episode {i}')
                t.set_postfix(
                    episode_reward=episode_reward, running_reward=running_reward
                )
                if running_reward >= target_reward:
                    break

        print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')


if __name__ == '__main__':
    en = create_gym_env('PongNoFrameskip-v4')[0]
    ac = A2C(en)
    ac.fit(18)
