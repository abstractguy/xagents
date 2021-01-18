import tensorflow as tf
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow_probability.python.distributions import Categorical


class CNNA2C(Model):
    def __init__(
        self,
        input_shape,
        n_actions,
        relu_gain=tf.math.sqrt(2.0),
        fc_units=512,
        actor_gain=0.01,
        critic_gain=1.0,
    ):
        relu_initializer = tf.initializers.Orthogonal(gain=relu_gain)
        super(CNNA2C, self).__init__()
        l1 = Conv2D(
            filters=32,
            input_shape=input_shape,
            kernel_size=8,
            strides=4,
            activation='relu',
            kernel_initializer=relu_initializer,
        )
        l2 = Conv2D(
            filters=64,
            kernel_size=4,
            activation='relu',
            strides=2,
            kernel_initializer=relu_initializer,
        )
        l3 = Conv2D(
            filters=64,
            kernel_size=3,
            activation='relu',
            strides=1,
            kernel_initializer=relu_initializer,
        )
        l4 = Flatten()
        l5 = Dense(fc_units, activation='relu', kernel_initializer=relu_initializer)
        self.common = Sequential([l1, l2, l3, l4, l5])
        self.critic = Dense(
            1,
            kernel_initializer=Orthogonal(critic_gain),
        )
        self.actor = Dense(n_actions, kernel_initializer=Orthogonal(gain=actor_gain))

    @tf.function
    def call(self, inputs, training=True, mask=None, actions=None):
        main = self.common(inputs, training=training)
        value = tf.squeeze(self.critic(main), axis=1)
        actor_features = self.actor(main)
        distribution = Categorical(logits=actor_features)
        if actions is None:
            actions = distribution.sample()
        action_log_probs = distribution.log_prob(actions)
        return (
            actions,
            action_log_probs,
            tf.reduce_mean(distribution.entropy()),
            value,
        )

    def get_config(self):
        super(CNNA2C, self).get_config()
