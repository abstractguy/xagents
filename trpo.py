from ppo import PPO


class TRPO(PPO):
    def __init__(self, envs, model, n_steps=512, lam=1, *args, **kwargs):
        super(TRPO, self).__init__(envs, model, n_steps, lam, *args, **kwargs)

    def train_step(self):
        pass


if __name__ == '__main__':
    from models import CNNA2C
    from utils import create_gym_env

    envi = create_gym_env('PongNoFrameskip-v4', 16)
    mod = CNNA2C(
        envi[0].observation_space.shape,
        envi[0].action_space.n,
    )
    agn = TRPO(envi, mod)
    agn.fit(19)
