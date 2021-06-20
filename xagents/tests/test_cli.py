from xagents.tests.utils import get_expected_flags


def test_help(executor, capsys, display_only_args):
    executor.maybe_create_agent(display_only_args[0])
    assert not executor.agent_id and not executor.command
    cap = capsys.readouterr().out
    for flag in display_only_args[1]:
        assert flag in cap


def test_non_help(executor, capsys, non_display_args):
    executor.maybe_create_agent(non_display_args)
    assert executor.agent_id and executor.command
    assert not capsys.readouterr().out


def test_parsers(executor, valid_parser_args):
    executor.command, executor.agent_id = valid_parser_args[:2]
    agent_args, non_agent_args, command_args = executor.parse_known_args(
        valid_parser_args
    )
    all_args = set()
    all_args.update(vars(agent_args).keys())
    all_args.update(vars(non_agent_args).keys())
    all_args.update(vars(command_args).keys())
    agent_expected_args = set(get_expected_flags(valid_parser_args, as_kwargs=True))
    assert all_args == agent_expected_args
