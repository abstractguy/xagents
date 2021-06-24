import random
from time import perf_counter

import gym
import numpy as np
import pytest
import tensorflow as tf
from gym.spaces import Discrete

import xagents
from xagents import A2C, ACER, DDPG, DQN, PPO, TD3, TRPO
from xagents.base import OffPolicy
from xagents.utils.buffers import ReplayBuffer


@pytest.mark.usefixtures('envs', 'model', 'buffers')
class TestBase:
    model_counts = {DDPG: 2, TD3: 3, TRPO: 2, A2C: 1, ACER: 1, DQN: 1, PPO: 1}

    def get_agent_kwargs(self, agent, envs=None, model=None, buffers=None):
        agent_kwargs = {'envs': envs if envs is not None else self.envs}
        buffers = buffers or self.buffers
        if self.model_counts[agent] > 1:
            agent_kwargs['actor_model'] = model or self.model
            agent_kwargs['critic_model'] = model or self.model
        else:
            agent_kwargs['model'] = model or self.model
        if agent == xagents.ACER:
            for buffer in buffers:
                buffer.batch_size = 1
            agent_kwargs['buffers'] = buffers
        if issubclass(agent, OffPolicy):
            agent_kwargs['buffers'] = buffers
        return agent_kwargs

    def test_no_envs(self, agent):
        with pytest.raises(AssertionError) as pe:
            agent_kwargs = self.get_agent_kwargs(agent, [])
            agent(**agent_kwargs)
        assert pe.value.args[0] == 'No environments given'

    def empty_buffers(self):
        for buffer in self.buffers:
            buffer.main_buffer.clear()

    def test_seeds(self, agent):
        test_seed = 1
        agent = agent(**self.get_agent_kwargs(agent))
        results1, results2 = set(), set()
        test_range = 1000000
        for results in [results1, results2]:
            agent.set_seeds(test_seed)
            results.add(random.randint(0, test_range))
            results.add(int(np.random.randint(0, test_range, 1)))
            results.add(int(tf.random.uniform([1], maxval=test_range)))
            results.add(self.envs[0].action_space.sample())
        assert results1 == results2

    @pytest.mark.parametrize(
        'env_id',
        [
            'RiverraidNoFrameskip-v4',
            'Robotank-ramDeterministic-v0',
            'BeamRiderDeterministic-v4',
            'PongNoFrameskip-v4',
            'GravitarNoFrameskip-v0',
            'Freeway-ramDeterministic-v0',
            'Bowling-ram-v0',
            'CentipedeDeterministic-v0',
            'Gopher-v4',
        ],
    )
    def test_n_actions(self, env_id, agent):
        env = gym.make(env_id)
        agent = agent(
            **self.get_agent_kwargs(agent, [env for _ in range(len(self.envs))])
        )
        if isinstance(env.action_space, Discrete):
            assert agent.n_actions == env.action_space.n
        else:
            assert agent.n_actions == env.action_space.shape

    def test_unsupported_env(self, agent):
        with pytest.raises(AssertionError) as pe:
            agent(
                **self.get_agent_kwargs(
                    agent, [gym.make('RepeatCopy-v0') for _ in range(len(self.envs))]
                )
            )
        assert 'Expected one of' in pe.value.args[0]

    def test_checkpoint(self, agent, tmp_path, capsys):
        checkpoints = {
            (tmp_path / f'test_weights{i}.tf').as_posix()
            for i in range(self.model_counts[agent])
        }
        expected_filenames = set()
        for checkpoint in checkpoints:
            expected_filenames.add(f'{checkpoint}.index')
            expected_filenames.add(f'{checkpoint}.data-00000-of-00001')
        expected_filenames.add((tmp_path / 'checkpoint').as_posix())
        agent = agent(**self.get_agent_kwargs(agent), checkpoints=checkpoints)
        agent.checkpoint()
        assert not capsys.readouterr().out
        agent.mean_reward = 100
        agent.checkpoint()
        assert 'Best reward updated' in capsys.readouterr().out
        assert agent.best_reward == 100
        resulting_files = {item.as_posix() for item in tmp_path.iterdir()}
        assert expected_filenames == resulting_files

    def test_wrong_checkpoints(self, agent):
        agent = agent(
            **self.get_agent_kwargs(agent),
            checkpoints=(self.model_counts[agent] + 1) * ['wrong_ckpt.tf'],
        )
        with pytest.raises(AssertionError) as pe:
            agent.fit(18)
        assert 'given output models, got' in pe.value.args[0]

    @staticmethod
    def check_progress_displayed(displayed):
        for item in (
            'time',
            'steps',
            'games',
            'speed',
            'mean reward',
            'best reward',
        ):
            assert item in displayed

    def test_display_metrics(self, capsys, agent):
        agent = agent(**self.get_agent_kwargs(agent))
        agent.training_start_time = perf_counter()
        agent.frame_speed = agent.mean_reward = agent.best_reward = 0
        agent.display_metrics()
        displayed = capsys.readouterr().out
        self.check_progress_displayed(displayed)

    def test_update_metrics(self, agent):
        agent = agent(**self.get_agent_kwargs(agent))
        agent.last_reset_time = perf_counter() - 2
        agent.steps = 1000
        agent.total_rewards.extend([100, 130, 150])
        agent.update_metrics()
        assert round(agent.frame_speed) == 500
        assert agent.mean_reward == 126.67

    def test_check_episodes(self, capsys, agent):
        agent = agent(**self.get_agent_kwargs(agent))
        agent.total_rewards.append(0)
        agent.training_start_time = perf_counter()
        agent.last_reset_time = perf_counter()
        agent.check_episodes()
        assert not capsys.readouterr().out
        agent.done_envs.extend(len(self.envs) * [1])
        agent.check_episodes()
        self.check_progress_displayed(capsys.readouterr().out)
        assert not agent.done_envs

    @pytest.mark.parametrize(
        'expected, mean_reward, steps, target_reward, max_steps',
        [
            (False, 10, 0, 20, None),
            (False, 10, 10, 11, 11),
            (True, 100, 1000, 90, 2000),
            (True, 200, 120, 500, 100),
            (True, 1, 1, 1, 1),
        ],
    )
    def test_training_done(
        self, capsys, expected, mean_reward, steps, target_reward, max_steps, agent
    ):
        agent = agent(**self.get_agent_kwargs(agent))
        agent.mean_reward = mean_reward
        agent.steps = steps
        agent.target_reward = target_reward
        agent.max_steps = max_steps
        status = agent.training_done()
        messages = ['Reward achieved in', 'Maximum steps exceeded']
        if not expected:
            assert not status and not capsys.readouterr().out
        else:
            assert status
            cap = capsys.readouterr().out
            assert any([message in cap for message in messages])

    def test_concat_buffer_samples(self, off_policy_agent):
        buffers = [ReplayBuffer(10, batch_size=1) for _ in range(4)]
        agent = off_policy_agent(
            **self.get_agent_kwargs(off_policy_agent, buffers=buffers)
        )
        with pytest.raises(ValueError) as pe:
            agent.concat_buffer_samples()
        assert 'Sample larger than population' in pe.value.args[0]
        seen = []
        for i, buffer in enumerate(agent.buffers):
            observation = [[i], [i * 10], [i * 100], [i * 1000]]
            seen.append(observation)
            buffer.append(*observation)
        expected = np.squeeze(np.array(seen, np.float32)).T
        result = agent.concat_buffer_samples()
        assert (expected == result).all()

    @pytest.mark.parametrize(
        'get_observation, store_in_buffers',
        [[False, False], [True, False], [False, True], [True, True]],
    )
    def test_step_envs(self, get_observation, store_in_buffers, agent):
        self.empty_buffers()
        agent = agent(**self.get_agent_kwargs(agent))
        actions = np.random.randint(0, agent.n_actions, len(self.envs))
        observations = agent.step_envs(actions, get_observation, store_in_buffers)
        if hasattr(agent, 'buffers'):
            if store_in_buffers:
                assert all([len(buffer.main_buffer) > 0 for buffer in agent.buffers])
            else:
                assert all([len(buffer.main_buffer) == 0 for buffer in agent.buffers])
        if get_observation:
            assert observations
            assert len(set([item.shape for item in observations[0]])) == 1
        else:
            assert not observations

    @pytest.mark.parametrize('target_reward, max_steps, monitor_session, checkpoints')
    def test_init_training(
        self, capsys, target_reward, max_steps, monitor_session, checkpoints
    ):
        pass
