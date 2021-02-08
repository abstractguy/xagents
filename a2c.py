import tensorflow as tf

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

    def step_transitions(self):
        transitions = states, actions, log_probs, values, rewards, masks, entropies = [
            [] for _ in range(7)
        ]
        step_states = tf.numpy_function(func=self.get_states, inp=[], Tout=tf.float32)
        for step in range(self.n_steps):
            step_actions, step_log_probs, step_entropies, step_values = self.model(
                step_states
            )
            step_states, step_rewards, step_dones = tf.numpy_function(
                func=self.step_envs,
                inp=[step_actions],
                Tout=(tf.float32, tf.float32, tf.float32),
            )
            step_masks = 1 - step_dones
            states.append(step_states)
            actions.append(step_actions)
            log_probs.append(step_log_probs)
            values.append(step_values)
            rewards.append(step_rewards)
            masks.append(step_masks)
            entropies.append(step_entropies)
        return transitions

    def calculate_loss(self, returns, values, log_probs, entropies):
        """
        Calculate total model loss.
        Args:
            returns: A list, the result of self.calculate_returns()
            values: list that will be the same size as self.n_steps and
                contains n step values and each step contains self.n_envs values.
            log_probs: list that will be the same size as self.n_steps and
                contains n step log_probs and each step contains self.n_envs log_probs.
            entropies: list that will be the same size as self.n_steps and
                contains n step entropies and each step contains self.n_envs entropies.
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
                actions,
                log_probs,
                values,
                rewards,
                masks,
                entropies,
            ) = self.step_transitions()
            next_values = self.model(states[-1])[-1]
            returns = [next_values]
            for step in reversed(range(self.n_steps)):
                returns.append(rewards[step] + masks[step] * self.gamma * returns[-1])
            returns.reverse()
            loss = self.calculate_loss(returns, values, log_probs, entropies)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, self.grad_norm)
        self.model.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )


if __name__ == '__main__':
    import tensorflow_addons as tfa

    from utils import create_gym_env

    ens = create_gym_env('PongNoFrameskip-v4', 16)
    from models import CNNA2C

    o = tfa.optimizers.RectifiedAdam(
        learning_rate=7e-4, epsilon=1e-5, beta_1=0.0, beta_2=0.99
    )
    m = CNNA2C(ens[0].observation_space.shape, ens[0].action_space.n)
    ac = A2C(ens, m, optimizer=o)
    ac.fit(19)
    # ac.play(
    #     '/Users/emadboctor/Desktop/code/drl-models/a2c-pong-17-model/a2c-pong.tf',
    #     render=True,
    # )
