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


def get_parser_args():
    argvs = []
    for command in xagents.commands:
        for agent in xagents.agents:
            argv = [command, agent, '--env', 'test-env']
            if command == 'train':
                argv.extend(['--target-reward', '-1'])
            argvs.append(argv)
    return argvs


def get_train_step_args():
    return [
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
        },
    ]
