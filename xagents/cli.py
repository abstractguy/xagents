import argparse
import sys

import pandas as pd

import xagents
from xagents import a2c, acer, dqn, ppo, td3, trpo
from xagents.utils.cli import general_args, play_args, train_args


def display_section(title, cli_args):
    section_frame = pd.DataFrame(cli_args).T.fillna('-')
    section_frame['flags'] = section_frame.index.values
    section_frame['flags'] = section_frame['flags'].apply(lambda flag: f'--{flag}')
    section_frame = section_frame.reset_index(drop=True).set_index('flags')
    print(f'\n{title.title()}\n')
    print(
        section_frame[
            [
                column_name
                for column_name in ('help', 'required', 'default')
                if column_name in section_frame.columns
            ]
        ].to_markdown()
    )


def display_commands(display_all=False):
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
    print('Use xagents <command> -h to see more info about a command', end='\n\n')
    print('Use xagents -h to display all command line options')
    if display_all:
        titles = 'train', 'play'
        cli_args = train_args, play_args
        for title, cli_arg in zip(titles, cli_args):
            display_section(title, cli_arg)


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


def train(cli_args):
    pass


def play(cli_args):
    pass


def execute():
    valid_commands = {'train': (train_args, train), 'play': (play_args, play)}
    agent_flags = {
        'a2c': a2c.cli_args,
        'acer': acer.cli_args,
        'dqn': dqn.cli_args,
        'ppo': ppo.cli_args,
        'td3': td3.cli_args,
        'trpo': trpo.cli_args,
    }
    if (total := len(cli_args := sys.argv)) == 1:
        display_commands()
        return
    if (command := cli_args[1]) in valid_commands and total == 2:
        display_section(command, valid_commands[command])
        return
    if (help_flags := any(('-h' in cli_args, '--help' in cli_args))) and total == 2:
        display_commands(True)
        return
    if total == 3 and command in valid_commands and help_flags:
        display_section(command, valid_commands[command])
        return
    if command not in valid_commands:
        print(f'Invalid command {command}')
        return
    parser = argparse.ArgumentParser()
    del sys.argv[1]
    add_args(general_args, parser)
    add_args(valid_commands[command][0], parser)
    process_args = parser.parse_args()
    selected_agent = process_args['algo']
    add_args(agent_flags[selected_agent], parser)
    all_args = parser.parse_args()
    valid_commands[command][1](all_args)


if __name__ == '__main__':
    execute()
