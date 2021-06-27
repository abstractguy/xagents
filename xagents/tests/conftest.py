import gym
import pytest
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

import xagents
from xagents import ACER, DDPG, DQN, TD3
from xagents.base import BaseAgent, OffPolicy, OnPolicy
from xagents.cli import Executor
from xagents.utils.buffers import ReplayBuffer
from xagents.utils.cli import agent_args, non_agent_args, play_args, train_args


@pytest.fixture(scope='function')
def executor(request):
    request.cls.executor = Executor()


@pytest.fixture(
    params=[
        ('non-agent', non_agent_args),
        ('agent', agent_args),
        ('train', train_args),
        ('play', play_args),
        ('do-nothing', {}),
    ]
)
def section(request):
    yield request.param


@pytest.fixture(scope='class')
def envs(request):
    request.cls.envs = [gym.make('PongNoFrameskip-v4') for _ in range(4)]


@pytest.fixture(scope='class')
def envs2(request):
    request.cls.envs2 = [gym.make('BipedalWalker-v3') for _ in range(4)]


@pytest.fixture(scope='class')
def model(request):
    x0 = Input((210, 160, 3))
    x = Dense(1, 'relu')(x0)
    x = Dense(1, 'relu')(x)
    x = Dense(1, 'relu')(x)
    model = request.cls.model = Model(x0, x)
    model.compile('adam')


@pytest.fixture(scope='class')
def buffers(request):
    request.cls.buffers = [ReplayBuffer(10, batch_size=155) for _ in range(4)]


@pytest.fixture(params=[item['agent'] for item in xagents.agents.values()])
def agent(request):
    yield request.param


@pytest.fixture(params=[agent_id for agent_id in xagents.agents])
def agent_id(request):
    yield request.param


@pytest.fixture(params=[command for command in xagents.commands])
def command(request):
    yield request.param


@pytest.fixture(params=[ACER, TD3, DQN, DDPG])
def off_policy_agent(request):
    yield request.param


@pytest.fixture(params=[BaseAgent, OnPolicy, OffPolicy])
def base_agent(request):
    yield request.param
