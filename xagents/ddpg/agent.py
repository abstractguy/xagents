import tensorflow as tf
from tensorflow.keras.losses import MSE

from xagents.base import OffPolicy


class DDPG(OffPolicy):
    def __init__(
        self,
        envs,
        actor_model,
        critic_model,
        buffers,
        gradient_steps=None,
        tau=0.05,
        step_noise_coef=0.1,
        **kwargs,
    ):
        super(DDPG, self).__init__(envs, actor_model, buffers, **kwargs)
        self.actor = actor_model
        self.critic1 = critic_model
        self.policy_delay = 1
        self.gradient_steps = gradient_steps
        self.tau = tau
        self.step_noise_coef = step_noise_coef
        self.episode_steps = tf.Variable(tf.zeros(self.n_envs), False)
        self.step_increment = tf.ones(self.n_envs)
        self.output_models = [self.actor, self.critic1]
        self.target_actor = tf.keras.models.clone_model(self.actor)
        self.target_critic1 = tf.keras.models.clone_model(self.critic1)
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic1.set_weights(self.critic1.get_weights())
        self.model_groups = [
            (self.actor, self.target_actor),
            (self.critic1, self.target_critic1),
        ]
        self.batch_dtypes = 5 * [tf.float32]

    def get_step_actions(self):
        actions = self.actor(tf.numpy_function(self.get_states, [], tf.float32))
        actions += tf.random.normal(
            shape=(self.n_envs, self.n_actions), stddev=self.step_noise_coef
        )
        return tf.clip_by_value(actions, -1, 1)

    def sync_target_models(self):
        """
        Sync target actor, target critic 1, and target critic 2 weights
        with their respective models in self.model_groups.

        Returns:
            None
        """
        for model, target_model in self.model_groups:
            for var, target_var in zip(
                model.trainable_variables, target_model.trainable_variables
            ):
                target_var.assign((1 - self.tau) * target_var + self.tau * var)

    def update_actor_weights(self, states):
        """
        Update actor weights.
        Args:
            states: A tensor of shape (self.n_envs * total buffer batch size, *self.input_shape)

        Returns:
            None.
        """
        with tf.GradientTape() as tape:
            actor_loss = -tf.reduce_mean(
                self.critic1(tf.concat([states, self.actor(states)], 1))
            )
        self.actor.optimizer.minimize(
            actor_loss, self.actor.trainable_variables, tape=tape
        )

    def update_critic_weights(self, states, actions, new_states, dones, rewards):
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(new_states)
            target_critic_input = tf.concat([new_states, target_actions], 1)
            target_value = self.target_critic1(target_critic_input)
            target_value = rewards + tf.stop_gradient(
                (1 - dones) * self.gamma * target_value
            )
            critic_input = tf.concat([states, actions], 1)
            value = self.critic1(critic_input)
            loss = MSE(value, target_value)
        self.critic1.optimizer.minimize(
            loss, self.critic1.trainable_variables, tape=tape
        )

    def update_weights(self, gradient_steps):
        """
        Run gradient steps and update both actor and critic weights according
            to self.policy delay for the given gradient steps.
        Args:
            gradient_steps: Number of iterations.

        Returns:
            None.
        """
        for gradient_step in range(int(gradient_steps)):
            states, actions, rewards, dones, new_states = tf.numpy_function(
                self.concat_buffer_samples, [], self.batch_dtypes
            )
            self.update_critic_weights(states, actions, new_states, dones, rewards)
            if gradient_step % self.policy_delay == 0:
                self.update_actor_weights(states)
                self.sync_target_models()

    @tf.function
    def train_step(self):
        """
        Perform 1 step which controls action_selection, interaction with environments
        in self.envs, batching and gradient updates.

        Returns:
            None
        """
        step_actions = self.get_step_actions()
        *_, dones, _ = tf.numpy_function(
            self.step_envs, [step_actions, True, True], self.batch_dtypes
        )
        for done_idx in tf.where(dones):
            gradient_steps = self.gradient_steps or self.episode_steps[done_idx[0]]
            self.update_weights(gradient_steps)
        self.episode_steps.assign(
            (self.episode_steps + self.step_increment) * (1 - dones)
        )
