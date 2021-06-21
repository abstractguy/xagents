import argparse
import sys
from pathlib import Path

import pandas as pd
from gym.spaces.discrete import Discrete
from tensorflow.keras.optimizers import Adam

import xagents
from xagents.base import OffPolicy
from xagents.utils.buffers import (
    IAmTheOtherKindOfReplayBufferBecauseFuckTensorflow,
    ReplayBuffer,
)
from xagents.utils.cli import agent_args, non_agent_args, off_policy_args
from xagents.utils.common import ModelReader, create_gym_env


class Executor:
    """
    Command line parser.
    """

    def __init__(self):
        """
        Initialize supported commands and agents.
        """
        self.valid_commands = xagents.commands
        self.available_agents = xagents.agents
        self.agent_id = None
        self.command = None
        self.agent = None

    @staticmethod
    def display_section(title, cli_args):
        """
        Display given title (command) and respective available options.
        Args:
            title: Command(s) that will be displayed on top of cli options.
            cli_args: A dictionary having flags and their respective
                `help`, `required` and `default`

        Returns:
            None
        """
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

    def display_commands(self, sections=None):
        """
        Display available commands and their description
            + command specific sections if given any.
        Args:
            sections: A dictionary having flags and their respective
                `help`, `required` and `default`

        Returns:
            None
        """
        print(f'xagents {xagents.__version__}')
        print(f'\nUsage:')
        print(f'\txagents <command> <agent> [options] [args]')
        print(f'\nAvailable commands:')
        for command, items in self.valid_commands.items():
            print(f'\t{command:<10} {items[2]}')
        print()
        print('Use xagents <command> to see more info about a command')
        print('Use xagents <command> <agent> to see more info about command + agent')
        if sections:
            for title, cli_args in sections.items():
                self.display_section(title, cli_args)

    @staticmethod
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
            _nargs = options.get('nargs')
            if not _action:
                parser.add_argument(
                    f'--{arg}',
                    help=_help,
                    default=_default,
                    type=_type,
                    required=_required,
                    nargs=_nargs,
                )
            else:
                parser.add_argument(
                    f'--{arg}', help=_help, default=_default, action=_action
                )

    def maybe_create_agent(self, cli_args):
        """
        Display help respective to parsed commands or set self.agent_id and self.command
        for further execution if enough arguments are given.

        Returns:
            None
        """
        to_display = {}
        total = len(cli_args)
        if total == 0:
            self.display_commands()
            return
        command = cli_args[0]
        to_display.update(non_agent_args)
        to_display.update(agent_args)
        assert command in self.valid_commands, f'Invalid command `{command}`'
        to_display.update(self.valid_commands[command][0])
        if total == 1:
            self.display_commands({command: to_display})
            return
        agent_id = cli_args[1]
        assert agent_id in self.available_agents, f'Invalid agent `{agent_id}`'
        to_display.update(self.available_agents[agent_id][0].cli_args)
        if total == 2:
            title = f'{command} {agent_id}'
            if (
                issubclass(self.available_agents[agent_id][1], OffPolicy)
                or agent_id == 'acer'
            ):
                to_display.update(off_policy_args)
            self.display_commands({title: to_display})
            return
        self.command, self.agent_id = command, agent_id

    def create_model(
        self,
        envs,
        agent_known_args,
        non_agent_known_args,
        model_suffix='',
        model_arg='model',
    ):
        """
        Create model using given .cfg path or use the respective agent's default.
        Args:
            envs: A list of gym environments.
            agent_known_args: kwargs passed to agent.
            non_agent_known_args: kwargs are general / not passed to agent.
            model_suffix: What follows cnn- or ann- and preceds extension in the default
                model configuration.
            model_arg: name of the kwarg found in `agent_known_args` which references
                the model configuration. If None is given, the default agent model
                configuration will be used.

        Returns:
            None
        """
        models_folder = (
            Path(self.available_agents[self.agent_id][0].__file__).parent / 'models'
        )
        units = [
            envs[0].action_space.n
            if isinstance(envs[0].action_space, Discrete)
            else envs[0].action_space.shape[0]
        ]
        if len(envs[0].observation_space.shape) == 3:
            network_type = 'cnn'
        else:
            network_type = 'ann'
        default_model_cfg = [*models_folder.rglob(f'{network_type}*{model_suffix}.cfg')]
        default_model_cfg = (
            default_model_cfg[0].as_posix() if default_model_cfg else None
        )
        model_cfg = agent_known_args[model_arg] or default_model_cfg
        assert (
            model_cfg
        ), f'You should specify --model <model.cfg>, no default model found in {models_folder}'
        if self.agent_id == 'acer':
            units.append(units[-1])
        elif 'actor' in model_cfg and 'critic' in model_cfg:
            units.append(1)
        elif 'critic' in model_cfg:
            units[0] = 1
        model_reader = ModelReader(
            model_cfg,
            units,
            envs[0].observation_space.shape,
            Adam(
                non_agent_known_args.lr,
                non_agent_known_args.beta1,
                non_agent_known_args.beta2,
                non_agent_known_args.opt_epsilon,
            ),
            agent_known_args['seed'],
        )
        if self.agent_id == 'td3' and 'critic' in model_cfg:
            model_reader.input_shape = (
                model_reader.input_shape[0] + envs[0].action_space.shape[0]
            )
        agent_known_args[model_arg] = model_reader.build_model()

    def parse_known_args(self, cli_args):
        """
        Parse general, agent and command specific args.

        Returns:
            agent kwargs, non-agent kwargs and command kwargs.
        """
        general_parser = argparse.ArgumentParser()
        agent_parser = argparse.ArgumentParser()
        command_parser = argparse.ArgumentParser()
        self.add_args(agent_args, agent_parser)
        self.add_args(self.available_agents[self.agent_id][0].cli_args, agent_parser)
        self.add_args(self.valid_commands[self.command][0], command_parser)
        if (
            issubclass(self.available_agents[self.agent_id][1], OffPolicy)
            or self.agent_id == 'acer'
        ):
            self.add_args(off_policy_args, general_parser)
        self.add_args(non_agent_args, general_parser)
        non_agent_known = general_parser.parse_known_args(cli_args)[0]
        agent_known = agent_parser.parse_known_args(cli_args)[0]
        command_known = command_parser.parse_known_args(cli_args)[0]
        if (
            self.command == 'train'
            and command_known.target_reward is None
            and command_known.max_steps is None
        ):
            command_parser.error('train requires --target-reward or --max-steps')
        return agent_known, non_agent_known, command_known

    def create_models(
        self,
        agent_known_args,
        non_agent_known_args,
        envs,
    ):
        """
        Create agent model(s).
        Args:
            agent_known_args: kwargs passed to agent.
            non_agent_known_args: kwargs are general / not passed to agent.
            envs: A list of gym environments.

        Returns:
            None
        """
        model_args = ['model', 'actor_model', 'critic_model']
        suffixes = ['', 'actor', 'critic']
        for model_arg, suffix in zip(model_args, suffixes):
            if model_arg in agent_known_args:
                self.create_model(
                    envs, agent_known_args, non_agent_known_args, suffix, model_arg
                )

    def create_buffers(self, agent_known_args, non_agent_known_args):
        """
        Create off-policy agent replay buffers.
        Args:
            agent_known_args: kwargs passed to agent.
            non_agent_known_args: kwargs are general / not passed to agent.

        Returns:
            None
        """
        buffer_max_size = non_agent_known_args.buffer_max_size // (
            non_agent_known_args.n_envs
        )
        buffer_initial_size = (
            non_agent_known_args.buffer_initial_size // non_agent_known_args.n_envs
            if non_agent_known_args.buffer_initial_size
            else buffer_max_size
        )
        buffer_batch_size = (
            non_agent_known_args.buffer_batch_size // non_agent_known_args.n_envs
        )
        if self.agent_id == 'td3':
            agent_known_args['buffers'] = [
                IAmTheOtherKindOfReplayBufferBecauseFuckTensorflow(
                    buffer_max_size,
                    5,
                    initial_size=buffer_initial_size,
                    batch_size=buffer_batch_size,
                )
                for _ in range(non_agent_known_args.n_envs)
            ]
        else:
            agent_known_args['buffers'] = [
                ReplayBuffer(
                    buffer_max_size,
                    non_agent_known_args.buffer_n_steps,
                    agent_known_args['gamma'],
                    initial_size=buffer_initial_size,
                    batch_size=buffer_batch_size,
                )
                for _ in range(non_agent_known_args.n_envs)
            ]

    def execute(self, cli_args=None):
        """
        Parse command line arguments, display help or execute command
        if enough arguments are given.

        Returns:
            None
        """
        self.maybe_create_agent(cli_args)
        if not self.agent_id:
            return
        agent_known, non_agent_known, command_known = self.parse_known_args(cli_args)
        envs = create_gym_env(
            non_agent_known.env,
            non_agent_known.n_envs,
            non_agent_known.preprocess,
            scale_frames=not non_agent_known.no_scale,
        )
        agent_known, command_known = vars(agent_known), vars(command_known)
        agent_known['envs'] = envs
        self.create_models(
            agent_known,
            non_agent_known,
            envs,
        )
        if (
            issubclass(self.available_agents[self.agent_id][1], OffPolicy)
            or self.agent_id == 'acer'
        ):
            self.create_buffers(agent_known, non_agent_known)
        self.agent = self.available_agents[self.agent_id][1](**agent_known)
        if non_agent_known.weights:
            n_weights = len(non_agent_known.weights)
            n_models = len(self.agent.output_models)
            assert (
                n_weights == n_models
            ), f'Expected {n_models} weights to load, got {n_weights} weights to load.'
            for weight, model in zip(non_agent_known.weights, self.agent.output_models):
                model.load_weights(weight).expect_partial()
        getattr(self.agent, self.valid_commands[self.command][1])(**command_known)


def execute(cli_args=None):
    cli_args = cli_args or sys.argv[1:]
    Executor().execute(cli_args)


if __name__ == '__main__':
    execute()
