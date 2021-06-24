import gym
import pytest
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

import xagents
from xagents import ACER, DDPG, DQN, TD3
from xagents.cli import Executor
from xagents.tests.utils import (get_display_cases, get_non_display_cases,
                                 get_parser_args, get_train_step_args)
from xagents.utils.buffers import ReplayBuffer


@pytest.fixture
def executor():
    return Executor()


@pytest.fixture(params=get_display_cases())
def display_only_args(request):
    yield request.param


@pytest.fixture(params=get_non_display_cases())
def non_display_args(request):
    yield request.param


@pytest.fixture(params=get_parser_args())
def parser_args(request):
    yield request.param


@pytest.fixture(params=get_train_step_args())
def train_args(request):
    yield request.param


@pytest.fixture(scope='class')
def envs(request):
    request.cls.envs = [gym.make('PongNoFrameskip-v4') for _ in range(4)]


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


@pytest.fixture(params=[item[1] for item in xagents.agents.values()])
def agent(request):
    yield request.param


@pytest.fixture(params=[ACER, TD3, DQN, DDPG])
def off_policy_agent(request):
    yield request.param
