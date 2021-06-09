cli_args = {
    'model': {'help': 'Path to model .cfg file'},
    'lam': {
        'help': 'GAE-Lambda for advantage estimation',
        'type': float,
        'default': 0.95,
    },
    'ppo-epochs': {
        'help': 'Gradient updates per training step',
        'type': int,
        'default': 4,
    },
    'mini-batches': {
        'help': 'Number of mini-batches to use per update',
        'type': int,
        'default': 4,
    },
    'advantage-epsilon': {
        'help': 'Value added to estimated advantage',
        'type': float,
        'default': 1e-8,
    },
    'clip-norm': {
        'help': 'Clipping value passed to tf.clip_by_value()',
        'type': float,
        'default': 0.1,
    },
}
