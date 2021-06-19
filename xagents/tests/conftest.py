import pytest

import xagents
from xagents.base import OffPolicy
from xagents.utils.cli import agent_args, non_agent_args, off_policy_args


def get_expected_kwargs(argv):
    if not argv:
        return []
    command = argv[0]
    expected_kwargs = {}
    expected_kwargs.update(agent_args)
    expected_kwargs.update(non_agent_args)
    expected_kwargs.update({command: xagents.commands[command]})
    if len(argv) > 1:
        agent_data = xagents.agents[argv[1]]
        expected_kwargs.update(agent_data[0].cli_args)
        if isinstance(agent_data[1], OffPolicy):
            expected_kwargs.update(off_policy_args)
    return expected_kwargs.keys()


def get_display_cases():
    valid_argvs = [([], [])]
    for command in xagents.commands:
        valid_argvs.append([[command], get_expected_kwargs([command])])
        for agent in xagents.agents:
            argv = [command, agent]
            valid_argvs.append((argv, get_expected_kwargs(argv)))
    return valid_argvs


@pytest.fixture(params=get_display_cases())
def display_only_args(request):
    yield request.param
