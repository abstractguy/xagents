import tensorflow as tf

from a2c import A2C
from utils import create_gym_env


class PPO(A2C):
    def __init__(
        self,
        envs,
        model,
        transition_steps=128,
        gae_lambda=0.95,
        ppo_epochs=4,
        mini_batches=4,
        advantage_epsilon=1e-5,
        clip_norm=0.1,
        *args,
        **kwargs,
    ):
        super(PPO, self).__init__(
            envs, model, transition_steps=transition_steps, *args, **kwargs
        )
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.mini_batches = mini_batches
        self.advantage_epsilon = advantage_epsilon
        self.clip_norm = clip_norm

    @tf.function
    def train_step(self):
        state_buffer = [tf.numpy_function(self.get_states, [], tf.float32)]
        rewards_buffer = []
        value_buffer = []
        returns_buffer = []
        lp_buffer = []
        action_buffer = []
        masks_buffer = []
        for step in range(self.transition_steps):
            actions, log_probs, _, values = self.model(state_buffer[step])
            states, rewards, dones = tf.numpy_function(
                self.step_envs, [actions], [tf.float32 for _ in range(3)]
            )
            masks = 1 - dones
            state_buffer.append(states)
            action_buffer.append(actions)
            lp_buffer.append(log_probs)
            value_buffer.append(values)
            rewards_buffer.append(rewards)
            masks_buffer.append(masks)
        *_, next_values = self.model(state_buffer[-1])
        value_buffer.append(next_values)
        gae = 0
        for step in reversed(range(self.transition_steps)):
            delta = (
                rewards_buffer[step]
                + self.gamma * value_buffer[step + 1] * masks_buffer[step] * gae
            )
            gae = delta + self.gamma * self.gae_lambda * masks_buffer[step] * gae
            returns_buffer.insert(0, gae + value_buffer[step])
        (state_b, rewards_b, value_b, returns_b, lp_b, action_b, masks_b) = [
            tf.concat(item, 0)
            for item in [
                state_buffer,
                rewards_buffer,
                value_buffer[:-1],
                returns_buffer,
                lp_buffer,
                action_buffer,
                masks_buffer,
            ]
        ]
        advantages = returns_b - value_b
        advantages = (advantages - tf.reduce_mean(advantages)) / (
            tf.math.reduce_std(advantages) + self.advantage_epsilon
        )
        batch_size = self.n_envs * self.transition_steps
        mini_batch_size = batch_size // self.mini_batches
        for epoch in range(self.ppo_epochs):
            indices = tf.random.shuffle(tf.range(batch_size))
            indices = tf.reshape(indices, (-1, mini_batch_size))
            for mini_batch_indices in indices:
                advantage_mb = tf.gather(
                    tf.reshape(advantages, (-1, 1)), mini_batch_indices
                )
                state_mb = tf.gather(
                    tf.reshape(state_b[:-1], (-1, *self.input_shape)),
                    mini_batch_indices,
                )
                action_mb = tf.gather(tf.reshape(action_b, (-1, 1)), mini_batch_indices)
                value_mb = tf.gather(
                    tf.reshape(value_b[:-1], (-1, 1)), mini_batch_indices
                )
                return_mb = tf.gather(
                    tf.reshape(returns_b, (-1, 1)), mini_batch_indices
                )
                lb_mb = tf.gather(tf.reshape(lp_b, (-1, 1)), mini_batch_indices)
                with tf.GradientTape() as tape:
                    _, mb_lb, mb_entropy, mb_value = self.model(
                        state_mb, actions=action_mb
                    )
                    mb_entropy = tf.reduce_mean(mb_entropy)
                    ratio = tf.exp(mb_lb - lb_mb)
                    s1 = ratio * advantage_mb
                    s2 = (
                        tf.clip_by_value(ratio, 1 - self.clip_norm, 1 + self.clip_norm)
                        * advantage_mb
                    )
                    action_loss = -tf.reduce_mean(tf.minimum(s1, s2))
                    value_mb_clipped = value_mb + tf.clip_by_value(
                        mb_value - value_mb, -self.clip_norm, self.clip_norm
                    )
                    value_loss = tf.square(mb_value - return_mb)
                    value_loss_clipped = tf.square(value_mb_clipped - return_mb)
                    value_loss = 0.5 * tf.reduce_mean(
                        tf.maximum(value_loss, value_loss_clipped)
                    )
                    loss = (
                        value_loss * self.value_loss_coef
                        + action_loss
                        - mb_entropy * self.entropy_coef
                    )
                gradients = tape.gradient(loss, self.model.trainable_variables)
                gradients, _ = tf.clip_by_global_norm(gradients, self.grad_norm)
                self.model.optimizer.apply_gradients(
                    zip(gradients, self.model.trainable_variables)
                )


if __name__ == '__main__':
    ens = create_gym_env('PongNoFrameskip-v4', 16)
    from tensorflow.keras.optimizers import Adam

    from models import CNNA2C

    m = CNNA2C(ens[0].observation_space.shape, ens[0].action_space.n)
    ac = PPO(ens, m, optimizer=Adam(2.5e-4, epsilon=1e-5))
    ac.fit(19)
