import tensorflow as tf
from tensorflow_probability.python.distributions import Categorical

from ppo import PPO


class TRPO(PPO):
    def __init__(
        self,
        envs,
        models,
        n_steps=512,
        lam=1,
        fvp_n_steps=5,
        entropy_coef=0,
        *args,
        **kwargs,
    ):
        super(TRPO, self).__init__(
            envs, models[0], n_steps, lam, entropy_coef=entropy_coef, *args, **kwargs
        )
        self.old_model = models[1]
        self.fvp_n_steps = fvp_n_steps

    def calculate_losses(self):
        pass

    def at_step_start(self):
        self.old_model.set_weights(self.model.get_weights())

    # @tf.function
    def train_step(self):
        states, actions, returns, values, log_probs = tf.numpy_function(
            self.get_batch, [], 5 * [tf.float32]
        )
        with tf.GradientTape(True) as tape:
            old_results = self.old_model(states)
            new_results = self.model(states)
            advantages = returns - values
            advantages = (advantages - tf.reduce_mean(advantages)) / tf.math.reduce_std(
                advantages
            )
            old_distribution = Categorical(old_results[-1])
            new_distribution = Categorical(new_results[-1])
            pass


if __name__ == '__main__':
    from utils import ModelHandler, create_gym_env

    mh = ModelHandler('models/cnn-ac.cfg', 6)
    m = mh.build_model()
    sd = None
    envi = create_gym_env('PongNoFrameskip-v4', 2)

    agn = TRPO(envi, m)
    agn.fit(19)
