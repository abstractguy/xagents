import random
from time import perf_counter

import gym
import numpy as np
import pytest
import tensorflow as tf
from gym.spaces import Discrete

from xagents.base import BaseAgent


@pytest.mark.usefixtures('envs', 'model', 'buffers')
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

    def test_n_actions(self):
        envs = [
            'RiverraidNoFrameskip-v4',
            'Robotank-ramDeterministic-v0',
            'BeamRiderDeterministic-v4',
            'PongNoFrameskip-v4',
            'GravitarNoFrameskip-v0',
            'Freeway-ramDeterministic-v0',
            'Bowling-ram-v0',
            'CentipedeDeterministic-v0',
            'Gopher-v4',
        ]
        for env_id in envs:
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

    def test_display_metrics(self, capsys):
        agent = BaseAgent(self.envs, self.model)
        agent.training_start_time = perf_counter()
        agent.frame_speed = agent.mean_reward = agent.best_reward = 0
        agent.display_metrics()
        displayed = capsys.readouterr().out
        for item in (
            'time',
            'steps',
            'games',
            'speed',
            'mean reward',
            'best reward',
        ):
            assert item in displayed
