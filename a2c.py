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
        """
        Initialize A2C agent.
        Args:
            envs: A list of gym environments.
            model: tf.keras.models.Model
            entropy_coef: Entropy coefficient used for entropy loss calculation.
            value_loss_coef: Value coefficient used for value loss calculation.
            transition_steps: n-step transition for example given s1, s2, s3, s4 and n_step = 4,
                transition will be s1 -> s4 (defaults to 1, s1 -> s2)
            *args: args Passed to BaseAgent.
            **kwargs: kwargs Passed to BaseAgent.
        """
        super(A2C, self).__init__(
            envs, model, *args, transition_steps=transition_steps, **kwargs
        )
        self.model = model
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef

    def get_states(self):
        """
        Get self.states
        Returns:
            self.states as numpy array
        """
        return np.array(self.states, np.float32)

    def calculate_returns(self, masks, rewards, values, log_probs, entropies):
        """
        Calculate returns to be used for loss calculation and gradient update.
        Args:
            masks: Empty list that will be the same size as self.transition_steps
                and will contain done masks.
            rewards: Empty list that will be the same size as self.transition_steps
                and will contain self.step_envs() rewards.
            values: Empty list that will be the same size as self.transition_steps
                and will contain self.step_envs() values.
            log_probs: Empty list that will be the same size as self.transition_steps
                and will contain self.step_envs() log_probs.
            entropies: Empty list that will be the same size as self.transition_steps
                and will contain self.step_envs() entropies.

        Returns:
            returns (a list that has most recent values after performing n steps)
        """
        states = tf.numpy_function(func=self.get_states, inp=[], Tout=tf.float32)
        for step in range(self.transition_steps):
            actions, step_log_probs, step_entropies, step_values = self.model(states)
            states, step_rewards, step_dones = tf.numpy_function(
                func=self.step_envs,
                inp=[actions],
                Tout=(tf.float32, tf.float32, tf.float32),
            )
            step_masks = 1 - step_dones
            log_probs.append(step_log_probs)
            values.append(step_values)
            rewards.append(step_rewards)
            masks.append(step_masks)
            entropies.append(step_entropies)
        next_values = self.model(states)[-1]
        returns = [next_values]
        for step in reversed(range(self.transition_steps)):
            returns.insert(0, rewards[step] + masks[step] * self.gamma * returns[0])
        return returns

    def calculate_loss(self, returns, values, log_probs, entropies):
        """
        Calculate total model loss.
        Args:
            returns: A list, the result of self.calculate_returns()
            values: list that will be the same size as self.transition_steps and
                contains n step values and each step contains self.n_envs values.
            log_probs: list that will be the same size as self.transition_steps and
                contains n step log_probs and each step contains self.n_envs log_probs.
            entropies: list that will be the same size as self.transition_steps and
                contains n step entropies and each step contains self.n_envs entropies.

        Returns:
            Total loss as tf.Tensor
        """
        value_loss = 0.0
        action_loss = 0.0
        entropy_loss = 0.0
        for step in range(self.transition_steps):
            advantages = tf.stop_gradient(returns[step]) - values[step]
            value_loss += tf.reduce_mean(tf.square(advantages))
            action_loss += -tf.reduce_mean(
                tf.stop_gradient(advantages) * log_probs[step]
            )
            entropy_loss += tf.reduce_mean(entropies[step])
        value_loss /= self.transition_steps
        action_loss /= self.transition_steps
        entropy_loss /= self.transition_steps
        return (
            self.value_loss_coef * value_loss
            + action_loss
            - entropy_loss * self.entropy_coef
        )

    @tf.function
    def train_step(self, clip_norm=0.5):
        """
        Do 1 training step.
        Args:
            clip_norm: Gradient clipping value passed to tf.clip_by_global_norm()

        Returns:
            None
        """
        masks = []
        rewards = []
        values = []
        log_probs = []
        entropies = []
        with tf.GradientTape() as tape:
            returns = self.calculate_returns(
                masks, rewards, values, log_probs, entropies
            )
            loss = self.calculate_loss(returns, values, log_probs, entropies)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)
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
        epsilon=1e-5,
        beta_1=0.0,
        beta_2=0.99,
    ):
        """
        Train agent on a supported environment
        Args:
            target_reward: Target reward, if achieved, the training will stop
            max_steps: Maximum number of steps, if reached the training will stop.
            monitor_session: Session name to use for monitoring the training with wandb.
            learning_rate: Model learning rate shared by both main and target networks.
            weights: Path to .tf trained model weights to continue training.
            epsilon: epsilon parameter passed to tfa.optimizers.RectifiedAdam()
            beta_1: beta_1 parameter passed tfa.optimizers.RectifiedAdam()
            beta_2: beta_2 parameter passed to tfa.optimizers.RectifiedAdam()
        Returns:
            None
        """
        optimizer = tfa.optimizers.RectifiedAdam(
            learning_rate=learning_rate, epsilon=epsilon, beta_1=beta_1, beta_2=beta_2
        )
        self.init_training(
            optimizer, target_reward, max_steps, monitor_session, weights, None
        )
        while True:
            self.check_episodes()
            if self.training_done():
                break
            self.train_step()


if __name__ == '__main__':
    ens = create_gym_env('PongNoFrameskip-v4', 16)
    from models import CNNA2C

    m = CNNA2C(ens[0].observation_space.shape, ens[0].action_space.n)
    ac = A2C(ens, m, checkpoint='a2c-pong.tf')
    # ac.fit(19, weights='a2c-pong.tf')
    ac.play(
        '/Users/emadboctor/Desktop/code/drl-models/a2c-pong-17-model/a2c-pong.tf',
        render=True,
    )
