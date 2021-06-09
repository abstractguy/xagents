import argparse
import sys
from pathlib import Path

import pandas as pd
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from tensorflow.keras.optimizers import Adam

import xagents
from xagents import (A2C, ACER, DQN, PPO, TD3, TRPO, a2c, acer, dqn, ppo, td3,
                     trpo)
from xagents.base import OffPolicy
from xagents.utils.buffers import (
    IAmTheOtherKindOfReplayBufferBecauseFuckTensorflow, ReplayBuffer)
from xagents.utils.cli import (agent_args, non_agent_args, off_policy_args,
                               play_args, train_args)
from xagents.utils.common import ModelReader, create_gym_env


def display_section(title, cli_args):
    section_frame = pd.DataFrame(cli_args).T.fillna('-')
    section_frame['flags'] = section_frame.index.values
    section_frame['flags'] = section_frame['flags'].apply(lambda flag: f'--{flag}')
    section_frame = section_frame.reset_index(drop=True).set_index('flags')
    print(f'\n{title}\n')
    print(
        section_frame[
            [
                column_name
                for column_name in ('help', 'required', 'default')
                if column_name in section_frame.columns
            ]
        ].to_markdown()
    )


def display_commands(sections=None):
    available_commands = {
        'train': 'Train given an agent and environment',
        'play': 'Play a game given a trained agent and environment',
    }
    print(f'xagents {xagents.__version__}')
    print(f'\nUsage:')
    print(f'\txagents <command> [options] [args]')
    print(f'\nAvailable commands:')
    for command, description in available_commands.items():
        print(f'\t{command:<10} {description}')
    print()
    print('Use xagents <command> to see more info about a command')
    print('Use xagents <command> <agent> to see more info about command + agent')
    if sections:
        for title, cli_args in sections.items():
            display_section(title, cli_args)


def add_args(cli_args, parser):
    """
    Add given arguments to parser.
    Args:
        cli_args: A dictionary of args and options.
        parser: argparse.ArgumentParser

    Returns:
        None.
    """
    for arg, options in cli_args.items():
        _help = options.get('help')
        _default = options.get('default')
        _type = options.get('type')
        _action = options.get('action')
        _required = options.get('required')
        if not _action:
            parser.add_argument(
                f'--{arg}', help=_help, default=_default, type=_type, required=_required
            )
        else:
            parser.add_argument(
                f'--{arg}', help=_help, default=_default, action=_action
            )


def train(cli_args, agent):
    envs = create_gym_env(cli_args.env, cli_args.n_envs, cli_args.preprocess)
    assert isinstance(
        envs[0].action_space, (Discrete, Box)
    ), f'Invalid env type {envs[0].action_space}'


def play(cli_args, agent):
    pass


def execute():
    valid_commands = {'train': (train_args, train), 'play': (play_args, play)}
    network_types = {Discrete: 'cnn', Box: 'ann'}
    agent_map = {
        'a2c': [a2c, A2C],
        'acer': [acer, ACER],
        'dqn': [dqn, DQN],
        'ppo': [ppo, PPO],
        'td3': [td3, TD3],
        'trpo': [trpo, TRPO],
    }
    if (total := len(cli_args := sys.argv)) == 1:
        display_commands()
        return
    if (command := cli_args[1]) not in valid_commands:
        print(f'Invalid command `{command}`')
        return
    if total == 2:
        display_commands({command: valid_commands[command][0]})
        return
    if total >= 3:
        agent_id = cli_args[2]
        if agent_id not in agent_map:
            print(f'Invalid agent `{agent_id}`')
            return
        if total == 3:
            title = f'{command} {agent_id}'
            to_display = valid_commands[command][0].copy()
            to_display.update(agent_args)
            to_display.update(non_agent_args)
            to_display.update(agent_map[agent_id][0])
            display_commands({title: to_display})
            return
        del sys.argv[1:3]
        general_parser = argparse.ArgumentParser()
        agent_parser = argparse.ArgumentParser()
        add_args(agent_args, agent_parser)
        add_args(agent_map[agent_id][0].cli_args, agent_parser)
        if issubclass(agent_map[agent_id][1], OffPolicy):
            add_args(off_policy_args, agent_parser)
        add_args(non_agent_args, general_parser)
        non_agent_known = general_parser.parse_known_args()[0]
        agent_known = vars(agent_parser.parse_known_args()[0])
        envs = create_gym_env(
            non_agent_known.env, non_agent_known.n_envs, non_agent_known.preprocess
        )
        agent_known['envs'] = envs
        if 'model' in agent_known:
            models_folder = Path(agent_map[agent_id][0].__file__).parent / 'models'
            network_type = network_types[type(envs[0].action_space)]
            default_model_cfg = [*models_folder.rglob(f'{network_type}*.cfg')]
            default_model_cfg = (
                default_model_cfg[0].as_posix() if default_model_cfg else None
            )
            assert (
                model_cfg := default_model_cfg or agent_known['model']
            ), f'You should specify --model <model.cfg>, no default model found in {models_folder}'
            units = [
                envs[0].action_space.n
                if isinstance(envs[0].action_space, Discrete)
                else envs[0].action_space.shape[0]
            ]
            if agent_id == 'acer':
                units.append(units[-1])
            elif 'actor' in model_cfg and 'critic' in model_cfg:
                units.append(1)
            elif 'critic' in model_cfg:
                units[0] = 1
            agent_known['model'] = ModelReader(
                model_cfg,
                units,
                envs[0].observation_space.shape,
                Adam(non_agent_known.lr),
                agent_known['seed'],
            ).build_model()
        if issubclass(agent_map[agent_id][1], OffPolicy) or agent_id == 'acer':
            buffer_max_size = non_agent_known.buffer_max_size // (
                n_envs := non_agent_known.n_envs
            )
            buffer_initial_size = non_agent_known.buffer_initial_size // n_envs
            buffer_batch_size = non_agent_known.buffer_batch_size // n_envs
            if agent_id == 'td3':
                agent_known['buffers'] = [
                    IAmTheOtherKindOfReplayBufferBecauseFuckTensorflow(
                        buffer_max_size,
                        5,
                        initial_size=buffer_initial_size,
                        batch_size=buffer_batch_size,
                    )
                    for _ in range(n_envs)
                ]
            else:
                agent_known['buffers'] = [
                    ReplayBuffer(
                        buffer_max_size,
                        non_agent_known.buffer_n_steps,
                        agent_known['gamma'],
                        initial_size=buffer_initial_size,
                        batch_size=buffer_batch_size,
                    )
                ]
        agent = agent_map[agent_id][1](**agent_known)
        pass


if __name__ == '__main__':
    execute()
