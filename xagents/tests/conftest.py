import gym
import pytest
from tensorflow.keras import Model

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
    request.cls.model = Model()


@pytest.fixture(scope='class')
def buffers(request):
    request.cls.buffers = [ReplayBuffer(1) for _ in range(4)]
