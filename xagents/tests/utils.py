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
        expected_kwargs.update(agent_data[0].cli_args)
        if issubclass(agent_data[1], OffPolicy) or argv[1] == 'acer':
            expected_kwargs.update(off_policy_args)
    if not as_kwargs:
        return expected_kwargs.keys()
    return [flag.replace('-', '_') for flag in expected_kwargs.keys()]


def get_display_cases():
    valid_argvs = [([], [])]
    for command in xagents.commands:
        valid_argvs.append([[command], get_expected_flags([command])])
        for agent in xagents.agents:
            argv = [command, agent]
            valid_argvs.append((argv, get_expected_flags(argv)))
    return valid_argvs


def get_non_display_cases():
    argvs = []
    for command in xagents.commands:
        for agent in xagents.agents:
            argv = [command, agent, '--env', 'test-env']
            argvs.append(argv)
    return argvs


def get_valid_parser_args():
    argvs = []
    for command in xagents.commands:
        for agent in xagents.agents:
            argv = [command, agent, '--env', 'test-env']
            if command == 'train':
                argv.extend(['--target-reward', '-1'])
            argvs.append(argv)
    return argvs
