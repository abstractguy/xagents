general_args = {
    'env': {'help': 'gym environment id', 'required': True},
    'algo': {'help': 'Algorithm to use', 'required': True}
}
train_args = {
    'reward-buffer-size': {
        'help': 'Size of the total reward buffer, used for calculating '
        'mean reward value to be displayed.',
        'default': 100,
        'type': int,
    },
    'n-envs': {'help': 'Number of environments to create', 'default': 1, 'type': int},
    'gamma': {'help': 'Discount factor', 'default': 0.99, 'type': float},
    'display-precision': {
        'help': 'Number of decimals to be displayed',
        'default': 2,
        'type': int,
    },
    'seed': {'help': 'Random seed', 'type': int},
    'scale-factor': {'help': 'Input scale divisor', 'type': int},
    'log-frequency': {'help': 'Log progress every n games', 'type': int},
    'preprocess': {
        'help': 'If specified, states will be treated as atari frames and preprocessed accordingly',
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
