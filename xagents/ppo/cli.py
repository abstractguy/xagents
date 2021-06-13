from xagents import a2c

a2c_args = a2c.cli_args
ppo_args = {
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
    'n-steps': {'help': 'Transition steps', 'type': int, 'default': 128},
}
cli_args = a2c_args.copy()
cli_args.update(ppo_args)
