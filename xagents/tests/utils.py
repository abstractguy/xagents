import xagents
from xagents.base import OffPolicy
from xagents.utils.cli import agent_args, non_agent_args, off_policy_args


def get_expected_flags(argv, as_kwargs=False):
    if not argv:
        return []
    command = argv[0]
    expected_kwargs = {}
    expected_kwargs.update(agent_args)
    expected_kwargs.update(non_agent_args)
    expected_kwargs.update(xagents.commands[command][0])
    if len(argv) > 1:
        agent_data = xagents.agents[argv[1]]
        expected_kwargs.update(agent_data['module'].cli_args)
        if issubclass(agent_data['agent'], OffPolicy) or argv[1] == 'acer':
            expected_kwargs.update(off_policy_args)
    if not as_kwargs:
        return expected_kwargs.keys()
    return [flag.replace('-', '_') for flag in expected_kwargs.keys()]


def check_displayed(cap, title, cli_args):
    assert title in cap
    for flag in cli_args:
        assert f'--{flag}' in cap
        for key in cli_args[flag]:
            if key in ['help', 'required', 'default']:
                for line in str(cli_args[flag][key]).split('\n'):
                    assert line in cap
