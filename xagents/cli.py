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


class Executor:
    def __init__(self):
        self.valid_commands = {
            'train': (train_args, 'fit', 'Train given an agent and environment'),
            'play': (
                play_args,
                'play',
                'Play a game given a trained agent and environment',
            ),
        }
        self.available_agents = {
            'a2c': [a2c, A2C],
            'acer': [acer, ACER],
            'dqn': [dqn, DQN],
            'ppo': [ppo, PPO],
            'td3': [td3, TD3],
            'trpo': [trpo, TRPO],
        }
        self.available_networks = {Discrete: 'cnn', Box: 'ann'}

    def display_section(self, title, cli_args):
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
        print(f'xagents {xagents.__version__}')
        print(f'\nUsage:')
        print(f'\txagents <command> [options] [args]')
        print(f'\nAvailable commands:')
        for command, items in self.valid_commands.items():
            print(f'\t{command:<10} {items[2]}')
        print()
        print('Use xagents <command> to see more info about a command')
        print('Use xagents <command> <agent> to see more info about command + agent')
        if sections:
            for title, cli_args in sections.items():
                self.display_section(title, cli_args)

    def add_args(self, cli_args, parser):
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
                    f'--{arg}',
                    help=_help,
                    default=_default,
                    type=_type,
                    required=_required,
                )
            else:
                parser.add_argument(
                    f'--{arg}', help=_help, default=_default, action=_action
                )

    def maybe_proceed(self):
        if (total := len(cli_args := sys.argv)) == 1:
            self.display_commands()
            return
        assert (
            command := cli_args[1]
        ) in self.valid_commands, f'Invalid command `{command}`'
        if total == 2:
            self.display_commands({command: self.valid_commands[command][0]})
            return
        assert (
            agent_id := cli_args[2]
        ) in self.available_agents, f'Invalid agent `{agent_id}`'
        if total == 3:
            title = f'{command} {agent_id}'
            to_display = self.valid_commands[command][0].copy()
            to_display.update(agent_args)
            to_display.update(non_agent_args)
            to_display.update(self.available_agents[agent_id][0].cli_args)
            self.display_commands({title: to_display})
            return
        return command, agent_id

    def create_model(
        self,
        agent_id,
        envs,
        agent_known_args,
        non_agent_known_args,
        model_suffix='',
        model_arg='model',
    ):
        models_folder = (
            Path(self.available_agents[agent_id][0].__file__).parent / 'models'
        )
        network_type = self.available_networks[type(envs[0].action_space)]
        units = [
            envs[0].action_space.n
            if isinstance(envs[0].action_space, Discrete)
            else envs[0].action_space.shape[0]
        ]
        default_model_cfg = [*models_folder.rglob(f'{network_type}*{model_suffix}.cfg')]
        default_model_cfg = (
            default_model_cfg[0].as_posix() if default_model_cfg else None
        )
        assert (
            model_cfg := agent_known_args[model_arg] or default_model_cfg
        ), f'You should specify --model <model.cfg>, no default model found in {models_folder}'
        if agent_id == 'acer':
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
        if agent_id == 'td3' and 'critic' in model_cfg:
            model_reader.input_shape = (
                model_reader.input_shape[0] + envs[0].action_space.shape[0]
            )
        agent_known_args[model_arg] = model_reader.build_model()

    def parse_known_args(self, command, agent_id):
        del sys.argv[1:3]
        general_parser = argparse.ArgumentParser()
        agent_parser = argparse.ArgumentParser()
        command_parser = argparse.ArgumentParser()
        self.add_args(agent_args, agent_parser)
        self.add_args(self.available_agents[agent_id][0].cli_args, agent_parser)
        self.add_args(self.valid_commands[command][0], command_parser)
        if issubclass(self.available_agents[agent_id][1], OffPolicy):
            self.add_args(off_policy_args, agent_parser)
        self.add_args(non_agent_args, general_parser)
        non_agent_known = general_parser.parse_known_args()[0]
        agent_known = vars(agent_parser.parse_known_args()[0])
        command_known = vars(command_parser.parse_known_args()[0])
        return agent_known, non_agent_known, command_known

    def create_models(
        self,
        agent_known_args,
        non_agent_known_args,
        agent_id,
        envs,
    ):
        if (model_arg := 'model') in agent_known_args:
            self.create_model(
                agent_id,
                envs,
                agent_known_args,
                non_agent_known_args,
                model_arg=model_arg,
            )
        if (model_arg := 'actor_model') in agent_known_args:
            self.create_model(
                agent_id,
                envs,
                agent_known_args,
                non_agent_known_args,
                'actor',
                model_arg,
            )
        if (model_arg := 'critic_model') in agent_known_args:
            self.create_model(
                agent_id,
                envs,
                agent_known_args,
                non_agent_known_args,
                'critic',
                model_arg,
            )

    def create_buffers(self, agent_known_args, non_agent_known_args, agent_id):
        buffer_max_size = non_agent_known_args.buffer_max_size // (
            n_envs := non_agent_known_args.n_envs
        )
        buffer_initial_size = (
            non_agent_known_args.buffer_initial_size // n_envs
            if non_agent_known_args.buffer_initial_size
            else buffer_max_size
        )
        buffer_batch_size = non_agent_known_args.buffer_batch_size // n_envs
        if agent_id == 'td3':
            agent_known_args['buffers'] = [
                IAmTheOtherKindOfReplayBufferBecauseFuckTensorflow(
                    buffer_max_size,
                    5,
                    initial_size=buffer_initial_size,
                    batch_size=buffer_batch_size,
                )
                for _ in range(n_envs)
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
                for _ in range(n_envs)
            ]

    def execute(self):
        if not (to_proceed_with := self.maybe_proceed()):
            return
        command, agent_id = to_proceed_with
        agent_known, non_agent_known, command_known = self.parse_known_args(
            command, agent_id
        )
        envs = create_gym_env(
            non_agent_known.env,
            non_agent_known.n_envs,
            non_agent_known.preprocess,
            scale_frames=not non_agent_known.no_scale,
        )
        agent_known['envs'] = envs
        self.create_models(
            agent_known,
            non_agent_known,
            agent_id,
            envs,
        )
        if (
            issubclass(self.available_agents[agent_id][1], OffPolicy)
            or agent_id == 'acer'
        ):
            self.create_buffers(agent_known, non_agent_known, agent_id)
        agent = self.available_agents[agent_id][1](**agent_known)
        if weights := non_agent_known.weights:
            assert (n_weights := len(weights)) == (
                n_models := len(agent.output_models)
            ), f'Expected {n_models} weights to load, got {n_weights} weights to load.'
            for weight, model in zip(weights, agent.output_models):
                model.load_weights(weight).expect_partial()
        getattr(agent, self.valid_commands[command][1])(**command_known)


if __name__ == '__main__':
    executor = Executor()
    executor.execute()
