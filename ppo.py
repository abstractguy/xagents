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
        *args,
        **kwargs,
    ):
        """
        Initialize PPO agent.
        Args:
            envs: A list of gym environments.
            model: tf.keras.models.Model
            *args: args Passed to BaseAgent.
            **kwargs: kwargs Passed to BaseAgent.
        """
        super(PPO, self).__init__(
            envs, model, transition_steps=transition_steps, *args, **kwargs
        )
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs

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
        states = self.step_transitions(log_probs, values, rewards, masks, entropies)
        next_values = self.model(states)[-1]
        values.append(next_values)
        gae = 0
        returns = []
        for step in reversed(range(self.transition_steps)):
            delta = (
                rewards[step]
                + self.gamma * values[step + 1] * masks[step]
                - values[step]
            )
            gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns


if __name__ == '__main__':
    ens = create_gym_env('PongNoFrameskip-v4', 16)
    from models import CNNA2C

    m = CNNA2C(ens[0].observation_space.shape, ens[0].action_space.n)
    ac = PPO(ens, m)
    ac.fit(19)
