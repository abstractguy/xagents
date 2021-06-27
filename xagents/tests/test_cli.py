import argparse
import random
import string

import pytest
import tensorflow as tf
from gym.spaces import Discrete

import xagents
from xagents.tests.utils import check_displayed, get_expected_flags


@pytest.mark.usefixtures('executor', 'envs', 'envs2')
class TestExecutor:
    def test_display_section(self, section, capsys):
        self.executor.display_section(*section)
        cap = capsys.readouterr().out
        check_displayed(cap, *section)

    def test_display_commands(self, section, capsys):
        self.executor.display_commands({section[0]: section[1]})
        cap = capsys.readouterr().out
        for keyword in [
            'xagents',
            'Usage',
            'xagents <command> <agent> [options] [args]',
            'Available commands:',
        ]:
            assert keyword in cap
        if section:
            check_displayed(cap, *section)

    @staticmethod
    def add_arg_value(_action, _type, _nargs, test_args, values, flag):
        if not _action:
            if _type in [int, float]:
                value = random.randint(0, 100)
            else:
                value = "".join(random.sample(string.ascii_lowercase, 10))
            if _nargs:
                value = [
                    "".join(random.sample(string.ascii_lowercase, 10))
                    for _ in range(10)
                ]
            if not isinstance(value, list):
                test_args.append(str(value))
                values[flag] = value
            else:
                for item in value:
                    test_args.append(f'{item}')
                values[flag] = value
        else:
            values[flag] = True

    def test_add_args(self, section):
        parser = argparse.ArgumentParser()
        self.executor.add_args(section[1], parser)
        test_args = []
        values = {}
        for flag, options in section[1].items():
            _help = options.get('help')
            _default = options.get('default')
            _type = options.get('type')
            _action = options.get('action')
            _nargs = options.get('nargs')
            test_args.append(f'--{flag}')
            self.add_arg_value(_action, _type, _nargs, test_args, values, flag)
        parsed_args = parser.parse_args(test_args)
        for attr, value in values.items():
            assert getattr(parsed_args, attr.replace('-', '_')) == value

    def test_maybe_create_agent_base_display(self, capsys):
        self.executor.maybe_create_agent([])
        cap = capsys.readouterr().out
        assert 'flag' not in cap
        for keyword in [
            'xagents',
            'Usage',
            'xagents <command> <agent> [options] [args]',
            'Available commands:',
        ]:
            assert keyword in cap

    def test_maybe_create_agent_invalid_command(self, capsys):
        with pytest.raises(AssertionError) as pe:
            self.executor.maybe_create_agent(['invalid'])
        assert 'Invalid command' in pe.value.args[0]

    def test_maybe_create_agent_command_agent(self, command, agent_id, capsys):
        test_args = [[command], [command, agent_id]]
        for argv in test_args:
            expected_flags = get_expected_flags(argv)
            self.executor.maybe_create_agent(argv)
            cap = capsys.readouterr().out
            for flag in expected_flags:
                assert f'--{flag}' in cap

    def test_maybe_create_agent_no_display(self, command, agent_id, capsys):
        argv = [command, agent_id, '--env', 'test-env']
        self.executor.maybe_create_agent(argv)
        assert not capsys.readouterr().out
        assert self.executor.agent_id == agent_id
        assert self.executor.command == command

    def test_parse_known_args(self, command, agent_id):
        self.executor.command = command
        self.executor.agent_id = agent_id
        argv = [command, agent_id, '--env', 'test-env']
        if command == 'train':
            argv.extend(['--target-reward', '18'])
        agent_args, non_agent_args, command_args = self.executor.parse_known_args(argv)
        actual = vars(agent_args) | vars(non_agent_args) | vars(command_args)
        assert set(get_expected_flags(argv, True)) == set(actual.keys())

    def test_create_model(self, agent_id):
        self.executor.command = 'train'
        self.executor.agent_id = agent_id
        argv = [
            self.executor.command,
            agent_id,
            '--env',
            'test-env',
            '--target-reward',
            '18',
        ]
        agent_args, non_agent_args, _ = self.executor.parse_known_args(argv)
        agent_args = vars(agent_args)
        envs = (
            self.envs if self.executor.agent_id not in ['td3', 'ddpg'] else self.envs2
        )
        agent_args['envs'] = envs
        expected_units = [
            envs[0].action_space.n
            if isinstance(envs[0].action_space, Discrete)
            else envs[0].action_space.shape[0]
        ]
        if agent_id == 'acer':
            expected_units.append(expected_units[-1])
        elif agent_id != 'dqn':
            expected_units.append(1)
        actual_units = []
        for model_arg in ['model', 'actor_model', 'critic_model']:
            if model_arg in agent_args:
                model = self.executor.create_model(
                    envs, agent_args, non_agent_args, model_arg
                )
                total_units = 2 if model_arg == 'model' and agent_id != 'dqn' else 1
                output_units = [
                    layer.units
                    for layer in model.layers
                    if isinstance(layer, tf.keras.layers.Dense)
                ][-total_units:]
                actual_units.extend(output_units)
        assert expected_units == actual_units

    @pytest.mark.parametrize(
        'train_args',
        [
            {
                'args': 'train a2c --env PongNoFrameskip-v4 --entropy-coef 5 '
                '--value-loss-coef 33 --grad-norm 7 --checkpoints xyz.tf '
                '--reward-buffer-size 150 --n-steps 100 --gamma 0.88 '
                '--display-precision 3 --seed 55 --scale-factor 50 '
                '--log-frequency 28 --n-envs 25 --lr 0.555 --opt-epsilon 0.3 '
                '--beta1 15 --beta2 12 --max-steps 1',
                'agent': {
                    'entropy_coef': 5,
                    'value_loss_coef': 33,
                    'grad_norm': 7,
                    'checkpoints': ['xyz.tf'],
                    'reward_buffer_size': 150,
                    'n_steps': 100,
                    'gamma': 0.88,
                    'display_precision': 3,
                    'seed': 55,
                    'scale_factor': 50,
                    'log_frequency': 28,
                    'n_envs': 25,
                },
                'non_agent': {
                    'env': 'PongNoFrameskip-v4',
                    'agent': xagents.A2C,
                    'lr': 0.555,
                    'opt_epsilon': 0.3,
                    'beta1': 15,
                    'beta2': 12,
                },
            }
        ],
    )
    def test_train(self, capsys, train_args):
        self.executor.execute(train_args['args'].split())
        assert 'Maximum steps exceeded' in capsys.readouterr().out
        for attr, value in train_args['agent'].items():
            assert getattr(self.executor.agent, attr) == value
        assert isinstance(self.executor.agent, train_args['non_agent']['agent'])
        assert (
            self.executor.agent.envs[0].unwrapped.spec.id
            == train_args['non_agent']['env']
        )
        assert (
            self.executor.agent.model.optimizer.learning_rate
            == train_args['non_agent']['lr']
        )
        assert (
            self.executor.agent.model.optimizer.epsilon
            == train_args['non_agent']['opt_epsilon']
        )
        assert (
            self.executor.agent.model.optimizer.beta_1
            == train_args['non_agent']['beta1']
        )
        assert (
            self.executor.agent.model.optimizer.beta_2
            == train_args['non_agent']['beta2']
        )
