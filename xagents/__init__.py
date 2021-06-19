from xagents import a2c, acer, dqn, ppo, td3, trpo
from xagents.a2c.agent import A2C
from xagents.acer.agent import ACER
from xagents.dqn.agent import DQN
from xagents.ppo.agent import PPO
from xagents.td3.agent import TD3
from xagents.trpo.agent import TRPO
from xagents.utils.cli import play_args, train_args

__author__ = 'Emad Boctor'
__email__ = 'emad_1989@hotmail.com'
__license__ = 'MIT'
__version__ = 1.0

agents = {
    'a2c': [a2c, A2C],
    'acer': [acer, ACER],
    'dqn': [dqn, DQN],
    'ppo': [ppo, PPO],
    'td3': [td3, TD3],
    'trpo': [trpo, TRPO],
}
commands = {
    'train': (train_args, 'fit', 'Train given an agent and environment'),
    'play': (
        play_args,
        'play',
        'Play a game given a trained agent and environment',
    ),
}
