import random
from time import perf_counter

import gym
import numpy as np
import pytest
import tensorflow as tf
from gym.spaces import Discrete

from xagents.base import BaseAgent, OffPolicy


@pytest.mark.usefixtures('envs', 'model')
class TestBase:
    def test_no_envs(self):
        with pytest.raises(AssertionError) as pe:
            args = [[], self.model]
            BaseAgent(*args)
            assert 'No environments given' in pe.value

    def test_seeds(self):
        test_seed = 1
        agent = BaseAgent(self.envs, self.model)
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
    def test_n_actions(self, env_id):
        env = gym.make(env_id)
        agent = BaseAgent([env], self.model)
        if isinstance(env.action_space, Discrete):
            assert agent.n_actions == env.action_space.n
        else:
            assert agent.n_actions == env.action_space.shape

    def test_unsupported_env(self):
        with pytest.raises(AssertionError) as pe:
            BaseAgent([gym.make('RepeatCopy-v0')], self.model)
            assert 'Expected one of' in pe.value

    def test_checkpoint(self, tmp_path, capsys):
        agent = BaseAgent(
            self.envs, self.model, checkpoints=[tmp_path / 'test_weights.tf']
        )
        agent.checkpoint()
        assert not capsys.readouterr().out
        agent.mean_reward = 100
        agent.checkpoint()
        assert 'Best reward updated' in capsys.readouterr().out
        assert agent.best_reward == 100
        assert {*tmp_path.iterdir()} == {
            tmp_path / item
            for item in (
                'checkpoint',
                'test_weights.tf.index',
                'test_weights.tf.data-00000-of-00001',
            )
        }

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

    def test_display_metrics(self, capsys):
        agent = BaseAgent(self.envs, self.model)
        agent.training_start_time = perf_counter()
        agent.frame_speed = agent.mean_reward = agent.best_reward = 0
        agent.display_metrics()
        displayed = capsys.readouterr().out
        self.check_progress_displayed(displayed)

    def test_update_metrics(self):
        agent = BaseAgent(self.envs, self.model)
        agent.last_reset_time = perf_counter() - 2
        agent.steps = 1000
        agent.total_rewards.extend([100, 130, 150])
        agent.update_metrics()
        assert round(agent.frame_speed) == 500
        assert agent.mean_reward == 126.67

    def test_check_episodes(self, capsys):
        agent = BaseAgent(self.envs, self.model)
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
        self, capsys, expected, mean_reward, steps, target_reward, max_steps
    ):
        agent = BaseAgent(self.envs, self.model)
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

    def test_concat_buffer_samples(self, buffers):
        envs = [gym.make('PongNoFrameskip-v4') for _ in range(4)]
        agent = OffPolicy(envs, self.model, buffers)
        with pytest.raises(ValueError) as pe:
            agent.concat_buffer_samples()
            assert 'Sample larger than population' in pe.value
        agent.set_seeds(1)
        for i, buffer in enumerate(agent.buffers):
            start = i * 10
            end = start + 10
            buffer.main_buffer.extend(
                [[n, n * 10, n * 100, n * 1000] for n in range(start, end)]
            )
        expected = [
            np.array([2.0, 1.0, 14.0, 11.0, 27.0, 29.0, 37.0, 36.0], dtype=np.float32),
            np.array(
                [20.0, 10.0, 140.0, 110.0, 270.0, 290.0, 370.0, 360.0], dtype=np.float32
            ),
            np.array(
                [200.0, 100.0, 1400.0, 1100.0, 2700.0, 2900.0, 3700.0, 3600.0],
                dtype=np.float32,
            ),
            np.array(
                [2000.0, 1000.0, 14000.0, 11000.0, 27000.0, 29000.0, 37000.0, 36000.0],
                dtype=np.float32,
            ),
        ]
        for expected_item, result in zip(expected, agent.concat_buffer_samples()):
            assert (expected_item == result).all()

    @pytest.mark.parametrize(
        'get_observation, store_in_buffers',
        [[False, False], [True, False], [False, True], [True, True]],
    )
    def test_step_envs(self, get_observation, store_in_buffers, buffers):
        agent = OffPolicy(self.envs, self.model, buffers)
        actions = np.random.randint(0, agent.n_actions, len(self.envs))
        observations = agent.step_envs(actions, get_observation, store_in_buffers)
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
