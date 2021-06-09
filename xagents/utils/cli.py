non_agent_args = {
    'env': {'help': 'gym environment id', 'required': True},
    'n-envs': {'help': 'Number of environments to create', 'default': 1, 'type': int},
    'preprocess': {
        'help': 'If specified, states will be treated as atari frames\n'
        'and preprocessed accordingly',
        'action': 'store_true',
    },
    'lr': {
        'help': 'Learning rate passed to a tensorflow.keras.optimizers.Optimizer',
        'type': float,
        'default': 7e-4,
    },
    'opt-epsilon': {
        'help': 'Epsilon passed to a tensorflow.keras.optimizers.Optimizer',
        'type': float,
        'default': 1e-7,
    },
    'beta1': {
        'help': 'Beta1 passed to a tensorflow.keras.optimizers.Optimizer',
        'type': float,
        'default': 0.9,
    },
    'beta2': {
        'help': 'Beta2 passed to a tensorflow.keras.optimizers.Optimizer',
        'type': float,
        'default': 0.999,
    },
    'buffer-max-size': {'help': 'Maximum replay buffer size', 'type': int, 'default': 50000},
    'buffer-initial-size': {'help': 'Replay buffer initial size', 'type': int, 'default': 10000},
    'buffer-batch-size': {
        'help': 'Replay buffer batch size',
        'type': int,
        'default': 32,
    },
    'buffer-n-steps': {
        'help': 'Replay buffer transition step',
        'type': int,
        'default': 1,
    },
}

agent_args = {
    'reward-buffer-size': {
        'help': 'Size of the total reward buffer, used for calculating\n'
        'mean reward value to be displayed.',
        'default': 100,
        'type': int,
    },
    'gamma': {'help': 'Discount factor', 'default': 0.99, 'type': float},
    'display-precision': {
        'help': 'Number of decimals to be displayed',
        'default': 2,
        'type': int,
    },
    'seed': {'help': 'Random seed', 'type': int},
    'scale-factor': {'help': 'Input scale divisor', 'type': int},
    'log-frequency': {'help': 'Log progress every n games', 'type': int},
}

off_policy_args = {
    'epsilon-start': {
        'help': 'Starting epsilon value which is used to control random exploration.\n'
        'It should be decremented and adjusted according to implementation needs',
        'type': float,
        'default': 1.0,
    },
    'epsilon-end': {
        'help': 'Epsilon end value (minimum exploration rate)',
        'type': float,
        'default': 0.02,
    },
    'epsilon-decay-steps': {
        'help': 'Number of steps for `epsilon-start` to reach `epsilon-end`',
        'type': float,
        'default': 150000,
    },
    'target-sync-steps': {
        'help': 'Sync target models every n steps',
        'type': int,
        'default': 1000,
    },
}

train_args = {
    'target-reward': {
        'help': 'Target reward when reached, training is stopped',
        type: int,
    },
    'max-steps': {
        'help': 'Maximum number of environment steps, when reached, training is stopped',
        type: int,
    },
    'monitor-session': {'help': 'Wandb session name'},
}

play_args = {
    'video-dir': {'help': 'Path to directory to save the resulting gameplay video'},
    'render': {
        'help': 'If specified, the gameplay will be rendered',
        'action': 'store_true',
    },
    'frame-dir': {'help': 'Path to directory to save game frames'},
    'frame-delay': {
        'help': 'Delay between rendered frames',
        'type': float,
        'default': 0,
    },
    'env-idx': {'help': 'env index in agent.envs', 'type': int, 'default': 0},
    'action-idx': {
        'help': 'Index of action output by agent.model',
        'type': int,
        'default': 0,
    },
}
