import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from base_agent import BaseAgent
from utils import create_gym_env


class A2C(BaseAgent):
    def __init__(
        self,
        envs,
        model,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        transition_steps=5,
        *args,
        **kwargs,
    ):
        super(A2C, self).__init__(
            envs, model, *args, transition_steps=transition_steps, **kwargs
        )
        self.model = model
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef

    def get_states(self):
        return np.array(self.states, np.float32)

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
                actions, log_prob, entropy, value = self.model(obs)
                obs, reward, done = tf.numpy_function(
                    func=self.step_envs,
                    inp=[actions],
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
        self.model.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )

    def fit(
        self,
        target_reward,
        max_steps=None,
        monitor_session=None,
        learning_rate=7e-4,
        weights=None,
    ):
        optimizer = tfa.optimizers.RectifiedAdam(
            learning_rate=learning_rate, epsilon=1e-5, beta_1=0.0, beta_2=0.99
        )
        self.init_training(
            optimizer, target_reward, max_steps, monitor_session, weights, None
        )
        while True:
            self.check_episodes()
            if self.training_done():
                break
            self.update()


if __name__ == '__main__':
    ens = create_gym_env('PongNoFrameskip-v4', 16)
    from models import CNNA2C

    m = CNNA2C(ens[0].observation_space.shape, ens[0].action_space.n)
    ac = A2C(ens, m)
    ac.fit(18)
