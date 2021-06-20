import pytest

from xagents.cli import Executor
from xagents.tests.utils import get_display_cases, get_non_display_cases, get_valid_parser_args


@pytest.fixture
def executor():
    return Executor()


@pytest.fixture(params=get_display_cases())
def display_only_args(request):
    yield request.param


@pytest.fixture(params=get_non_display_cases())
def non_display_args(request):
    yield request.param


@pytest.fixture(params=get_valid_parser_args())
def valid_parser_args(request):
    yield request.param
