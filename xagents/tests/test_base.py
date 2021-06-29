import random
from time import perf_counter

import gym
import numpy as np
import pytest
import tensorflow as tf
import wandb
from gym.spaces import Discrete

import xagents
from xagents import A2C, ACER, DDPG, DQN, PPO, TD3, TRPO
from xagents.base import BaseAgent, OffPolicy, OnPolicy
from xagents.utils.buffers import ReplayBuffer
from xagents.utils.common import get_wandb_key


@pytest.mark.usefixtures('envs', 'envs2', 'model', 'buffers', 'executor')
class TestBase:
    """
    Tests for base agents and their methods.
    """

    model_counts = {
        DDPG: 2,
        TD3: 3,
        TRPO: 2,
        A2C: 1,
        ACER: 1,
        DQN: 1,
        PPO: 1,
        BaseAgent: 1,
        OnPolicy: 1,
        OffPolicy: 1,
    }

    def get_agent_kwargs(
        self, agent, envs=None, model=None, buffers=None, critic_model=None
    ):
        """
        Construct agent required kwargs with according to the given agent.
        Args:
            agent: OnPolicy/OffPolicy subclass.
            envs: A list of gym environments, if not specified, the default
                class envs will be returned.
            model: tf.keras.Model, if not specified, the default class model
                will be returned.
            buffers: A list of replay buffers, if not specified, the default
                class buffers will be returned.
            critic_model: tf.keras.Model, if not specified, the default class
                model will be returned.

        Returns:
            agent_kwargs.
        """
        agent_kwargs = {'envs': envs if envs is not None else self.envs}
        buffers = buffers or self.buffers
        if self.model_counts[agent] > 1:
            agent_kwargs['actor_model'] = model or self.model
            agent_kwargs['critic_model'] = critic_model or self.model
        else:
            agent_kwargs['model'] = model or self.model
        if agent == xagents.ACER:
            for buffer in buffers:
                buffer.batch_size = 1
            agent_kwargs['buffers'] = buffers
        if issubclass(agent, OffPolicy) or agent == OffPolicy:
            agent_kwargs['buffers'] = buffers
        return agent_kwargs

    def test_no_envs(self, agent):
        """
        Test no given environments.
        Args:
            agent: OnPolicy/OffPolicy subclass.
        """
        with pytest.raises(AssertionError) as pe:
            agent_kwargs = self.get_agent_kwargs(agent, [])
            agent(**agent_kwargs)
        assert pe.value.args[0] == 'No environments given'

    def test_seeds(self, agent):
        """
        Test random seeds (random/numpy/tensorflow/gym)
        Args:
            agent: OnPolicy/OffPolicy subclass.
        """
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
        """
        Test if agent initializes the correct n_actions attribute.
        Args:
            env_id: One of agent ids available in xagents.agents
            agent: OnPolicy/OffPolicy subclass.
        """
        env = gym.make(env_id)
        agent = agent(
            **self.get_agent_kwargs(agent, [env for _ in range(len(self.envs))])
        )
        if isinstance(env.action_space, Discrete):
            assert agent.n_actions == env.action_space.n
        else:
            assert agent.n_actions == env.action_space.shape

    def test_unsupported_env(self, agent):
        """
        Test currently unsupported environments including multi-discrete.
        Args:
            agent: OnPolicy/OffPolicy subclass.
        """
        with pytest.raises(AssertionError) as pe:
            agent(
                **self.get_agent_kwargs(
                    agent, [gym.make('RepeatCopy-v0') for _ in range(len(self.envs))]
                )
            )
        assert 'Expected one of' in pe.value.args[0]

    def test_checkpoint(self, agent, tmp_path, capsys):
        """
        Test display after keywords that are expected after a checkpoint.
        Also test for the presence of the expected checkpoint .tf files
        that are usually saved during training and ensure the number of models
        per agent in self.model_counts matches the one required by the agent,
        otherwise, an error will be raised by the agent.
        Args:
            agent: OnPolicy/OffPolicy subclass.
            tmp_path: pathlib.PosixPath
            capsys: _pytest.capture.CaptureFixture
        """
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
        """
        Ensure an error is raised for the wrong number of checkpoints.
        Args:
            agent: OnPolicy/OffPolicy subclass.
        """
        agent = agent(
            **self.get_agent_kwargs(agent),
            checkpoints=(self.model_counts[agent] + 1) * ['wrong_ckpt.tf'],
        )
        with pytest.raises(AssertionError) as pe:
            agent.fit(18)
        assert 'given output models, got' in pe.value.args[0]

    @staticmethod
    def check_progress_displayed(displayed):
        """
        Check text displayed in the console and ensure it has
        the expected keywords.
        Args:
            displayed: str
        """
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
        """
        Check metric names are printed to the console.
        Args:
            capsys: _pytest.capture.CaptureFixture
            agent: OnPolicy/OffPolicy subclass.
        """
        agent = agent(**self.get_agent_kwargs(agent))
        agent.training_start_time = perf_counter()
        agent.frame_speed = agent.mean_reward = agent.best_reward = 0
        agent.display_metrics()
        displayed = capsys.readouterr().out
        self.check_progress_displayed(displayed)

    def test_update_metrics(self, agent):
        """
        Ensure accuracy of the metrics displayed as the training
            progresses.
        Args:
            agent: OnPolicy/OffPolicy subclass.
        """
        agent = agent(**self.get_agent_kwargs(agent))
        agent.last_reset_time = perf_counter() - 2
        agent.steps = 1000
        agent.total_rewards.extend([100, 130, 150])
        agent.update_metrics()
        assert round(agent.frame_speed) == 500
        assert agent.mean_reward == 126.67

    def test_check_episodes(self, capsys, agent):
        """
        Ensure progress is displayed when expected and flags
            are updated afterwards.
        Args:
            capsys: _pytest.capture.CaptureFixture
            agent: OnPolicy/OffPolicy subclass.
        """
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
        """
        Test training status (done / not done) and ensure respective expected
            actions take place.
        Args:
            capsys: _pytest.capture.CaptureFixture
            expected: Expected training status where True indicates training
                should be done.
            mean_reward: Agent `mean_reward` attribute to set.
            steps: Agent `steps` attribute to set
            target_reward: Agent `target_reward` attribute to set.
            max_steps: Agent `max_steps` attribute to set
            agent: OnPolicy/OffPolicy subclass.
        """
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
        """
        Test multiple buffer samples are concatenated properly.
        Args:
            off_policy_agent: OffPolicy subclass.
        """
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
        """
        Test observations are returned / stored in buffers when this is expected.
        Args:
            get_observation: arg passed to step_envs()
            store_in_buffers: arg passed to step_envs()
            agent: OnPolicy/OffPolicy subclass.
        """
        for buffer in self.buffers:
            buffer.main_buffer.clear()
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

    @pytest.mark.skipif(not get_wandb_key(), reason='Wandb api key not available')
    @pytest.mark.parametrize(
        'target_reward, max_steps, monitor_session',
        [[18, None, None], [22, 55, None], [None, 100, 'test_session']],
    )
    def test_init_training(
        self, agent, capsys, target_reward, max_steps, monitor_session, tmpdir
    ):
        """
        Test training initialization and ensure training flags are set and
        a monitoring session is registered if one is specified.
        Args:
            agent: OnPolicy/OffPolicy subclass.
            capsys: _pytest.capture.CaptureFixture
            target_reward: arg passed to init_training()
            max_steps: arg passed to init_training()
            monitor_session: arg passed to init_training()
            tmpdir: py._path.local.LocalPath
        """
        checkpoints = [f'test-{i}.tf' for i in range(self.model_counts[agent])]
        agent = agent(**self.get_agent_kwargs(agent))
        agent.checkpoints = checkpoints
        with tmpdir.as_cwd():
            agent.init_training(target_reward, max_steps, monitor_session)
        assert agent.target_reward == target_reward
        assert agent.max_steps == max_steps
        assert agent.last_reset_time
        assert agent.training_start_time
        if monitor_session:
            assert wandb.run.name == monitor_session
            capsys.readouterr()

    def test_train_step(self, base_agent):
        """
        Ensure an exception is raised when train_step() abstract method is called.
        Args:
            base_agent: A base agent class.
        """
        agent = base_agent(**self.get_agent_kwargs(base_agent))
        with pytest.raises(NotImplementedError) as pe:
            agent.train_step()
        assert 'train_step() should be implemented by' in pe.value.args[0]

    def test_get_model_inputs(self, agent):
        """
        Validate `scale_factor` arg passed to agent and ensure
        states are scaled properly.
        Args:
            agent: OnPolicy/OffPolicy subclass.
        """
        inputs = np.random.random((10, 10))
        scale_factor = 122
        agent_kwargs = self.get_agent_kwargs(agent)
        with pytest.raises(AssertionError) as pe:
            agent(**agent_kwargs, scale_factor=0)
        assert 'Invalid scale factor' in pe.value.args[0]
        agent = agent(**agent_kwargs, scale_factor=scale_factor)
        if scale_factor:
            expected = inputs / scale_factor
            actual = agent.get_model_inputs(inputs)
            assert np.isclose(expected, actual).all()

    def test_get_model_outputs(self, base_agent):
        """
        Test single and multiple model outputs.
        Args:
            base_agent: A base agent class.
        """
        inputs = np.random.random((10, *self.envs[0].observation_space.shape))
        agent = base_agent(**self.get_agent_kwargs(base_agent))
        single_model_result = agent.get_model_outputs(inputs, self.model)
        multi_model_result = agent.get_model_outputs(inputs, [self.model, self.model])
        assert single_model_result.shape == multi_model_result[0].shape
        assert np.isclose(
            single_model_result.numpy(), multi_model_result[0].numpy()
        ).all()

    def test_play(
        self,
        agent,
        capsys,
        tmp_path,
        max_steps=10,
    ):
        """
        Test 1 game play and ensure resulting video / frames are saved.
        Args:
            agent: OnPolicy/OffPolicy subclass.
            capsys: _pytest.capture.CaptureFixture
            max_steps: arg passed to init_training()
            tmp_path: pathlib.PosixPath
        """
        agent_kwargs = {}
        agent_id = agent.__module__.split('.')[1]
        envs = agent_kwargs['envs'] = (
            self.envs if agent_id not in ['td3', 'ddpg'] else self.envs2
        )
        self.executor.agent_id = agent_id
        self.executor.command = 'play'
        agent_args, non_agent_args, command_args = self.executor.parse_known_args(
            [
                'play',
                agent_id,
                '--env',
                envs[0].spec.id,
                '--buffer-batch-size',
                '1000',
                '--n-envs',
                f'{len(self.envs)}',
            ]
        )
        agent_args, command_args = vars(agent_args), vars(command_args)
        agent_args['envs'] = envs
        models = self.executor.create_models(
            agent_args,
            non_agent_args,
            envs,
        )
        if issubclass(agent, OffPolicy) or agent == ACER:
            agent_kwargs['buffers'] = self.executor.create_buffers(
                agent_args, non_agent_args
            )
        if len(models) == 1:
            agent_kwargs['model'] = models[0]
        else:
            agent_kwargs['actor_model'] = models[0]
            agent_kwargs['critic_model'] = models[1]
        agent = agent(**agent_kwargs)
        video_dir = tmp_path / agent_id / 'video'
        frame_dir = tmp_path / agent_id / 'frames'
        agent.play(video_dir, False, frame_dir, 0, max_steps)
        expected_frames = [frame_dir / f'{i:05d}.jpg' for i in range(max_steps)]
        for expected_frame in expected_frames:
            assert expected_frame.exists()
        assert len([*video_dir.rglob('openai*.mp4')])
        assert 'Maximum steps' in capsys.readouterr().out
