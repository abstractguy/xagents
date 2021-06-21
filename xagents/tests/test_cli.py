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


def test_parsers(executor, parser_args):
    executor.command, executor.agent_id = parser_args[:2]
    agent_args, non_agent_args, command_args = executor.parse_known_args(parser_args)
    all_args = set()
    all_args.update(vars(agent_args).keys())
    all_args.update(vars(non_agent_args).keys())
    all_args.update(vars(command_args).keys())
    agent_expected_args = set(get_expected_flags(parser_args, as_kwargs=True))
    assert all_args == agent_expected_args


def test_train(executor, train_step_args, capsys):
    executor.execute(train_step_args['args'].split())
    assert 'Maximum steps exceeded' in capsys.readouterr().out
    for attr, value in train_step_args['agent'].items():
        assert getattr(executor.agent, attr) == value
    assert isinstance(executor.agent, train_step_args['non_agent']['agent'])
    assert (
        executor.agent.envs[0].unwrapped.spec.id == train_step_args['non_agent']['env']
    )
    assert (
        executor.agent.model.optimizer.learning_rate
        == train_step_args['non_agent']['lr']
    )
    assert (
        executor.agent.model.optimizer.epsilon
        == train_step_args['non_agent']['opt_epsilon']
    )
    assert (
        executor.agent.model.optimizer.beta_1 == train_step_args['non_agent']['beta1']
    )
    assert (
        executor.agent.model.optimizer.beta_2 == train_step_args['non_agent']['beta2']
    )
