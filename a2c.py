import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions import Categorical

from base_agent import BaseAgent


class A2C(BaseAgent):
    def __init__(
        self,
        envs,
        model,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        n_steps=5,
        grad_norm=0.5,
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
            n_steps: n-step transition for example given s1, s2, s3, s4 and n_step = 4,
                transition will be s1 -> s4 (defaults to 1, s1 -> s2)
            grad_norm: Gradient clipping value passed to tf.clip_by_global_norm()
            *args: args Passed to BaseAgent.
            **kwargs: kwargs Passed to BaseAgent.
        """
        super(A2C, self).__init__(envs, model, *args, n_steps=n_steps, **kwargs)
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.grad_norm = grad_norm
        activations = [layer.activation for layer in model.layers[-2:]]
        self.output_is_softmax = tf.keras.activations.softmax in activations
        self.critic_units = model.layers[-1].units

    def get_distribution(self, actor_output):
        if self.output_is_softmax:
            return Categorical(probs=actor_output)
        return Categorical(logits=actor_output)

    @tf.function
    def get_model_outputs(self, inputs, model, training=True, actions=None):
        inputs = self.get_model_inputs(inputs)
        actor_output, critic_output = model(inputs, training=training)
        distribution = self.get_distribution(actor_output)
        if self.critic_units == 1:
            critic_output = tf.squeeze(critic_output)
        if actions is None:
            actions = distribution.sample(seed=self.seed)
        action_log_probs = distribution.log_prob(actions)
        return (
            actions,
            action_log_probs,
            critic_output,
            distribution.entropy(),
            actor_output,
        )

    def get_batch(self):
        """
        Get n-step batch which is the result of running self.envs step() for
        self.n_steps times.

        Returns:
            A list of numpy arrays which contains
             [states, rewards, actions, critic_output, dones, log probs, entropies, actor_output]
        """
        batch = (
            states,
            rewards,
            actions,
            critic_output,
            dones,
            log_probs,
            entropies,
            actor_output,
        ) = [[] for _ in range(8)]
        step_states = tf.numpy_function(self.get_states, [], tf.float32)
        step_dones = tf.numpy_function(self.get_dones, [], tf.float32)
        for _ in range(self.n_steps):
            (
                step_actions,
                step_log_probs,
                step_values,
                step_entropies,
                step_actor_logits,
            ) = self.get_model_outputs(step_states, self.model)
            states.append(step_states)
            actions.append(step_actions)
            critic_output.append(step_values)
            log_probs.append(step_log_probs)
            dones.append(step_dones)
            entropies.append(step_entropies)
            actor_output.append(step_actor_logits)
            *_, step_rewards, step_dones, step_states = tf.numpy_function(
                self.step_envs,
                [step_actions, True, False],
                5 * [tf.float32],
            )
            rewards.append(step_rewards)
        dones.append(step_dones)
        return batch

    def calculate_loss(
        self,
        returns,
        values,
        entropies,
        log_probs,
    ):
        """
        Calculate total model loss.
        Args:
            returns: A list, the result of self.calculate_returns()
            values: list that will be the same size as self.n_steps and
                contains n step values and each step contains self.n_envs values.
            entropies: list that will be the same size as self.n_steps and
                contains n step entropies and each step contains self.n_envs entropies.
            log_probs: list that will be the same size as self.n_steps and
                contains n step log_probs and each step contains self.n_envs log_probs.

        Returns:
            Total loss as tf.Tensor
        """
        value_loss = 0.0
        action_loss = 0.0
        entropy_loss = 0.0
        for step in range(self.n_steps):
            advantages = tf.stop_gradient(returns[step]) - values[step]
            value_loss += tf.reduce_mean(tf.square(advantages))
            action_loss += -tf.reduce_mean(
                tf.stop_gradient(advantages) * log_probs[step]
            )
            entropy_loss += tf.reduce_mean(entropies[step])
        value_loss /= self.n_steps
        action_loss /= self.n_steps
        entropy_loss /= self.n_steps
        return (
            self.value_loss_coef * value_loss
            + action_loss
            - entropy_loss * self.entropy_coef
        )

    @tf.function
    def train_step(self):
        """
        Do 1 training step.

        Returns:
            None
        """
        with tf.GradientTape() as tape:
            (
                states,
                rewards,
                actions,
                values,
                dones,
                log_probs,
                entropies,
                _,
            ) = self.get_batch()
            masks = 1 - np.array(dones)
            next_values = self.get_model_outputs(states[-1], self.model)[2]
            returns = [next_values]
            for step in reversed(range(self.n_steps)):
                returns.append(
                    rewards[step] + masks[step + 1] * self.gamma * returns[-1]
                )
            returns.reverse()
            loss = self.calculate_loss(returns, values, entropies, log_probs)
        grads = tape.gradient(loss, self.model.trainable_variables)
        if self.grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.grad_norm)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


if __name__ == '__main__':
    import tensorflow_addons as tfa

    from utils import ModelHandler, create_gym_env

    ens = create_gym_env('PongNoFrameskip-v4', 16)

    o = tfa.optimizers.RectifiedAdam(
        learning_rate=7e-4, epsilon=1e-5, beta_1=0.0, beta_2=0.99
    )
    mh = ModelHandler('models/cnn-ac.cfg', [ens[0].action_space.n, 1])
    m = mh.build_model()
    ac = A2C(ens, m, optimizer=o)
    ac.fit(19)
    # ac.play(
    #     '/Users/emadboctor/Desktop/code/drl-models/a2c-pong-17-model/a2c-pong.tf',
    #     render=True,
    # )
