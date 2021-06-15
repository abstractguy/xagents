cli_args = {
    'model': {'help': 'Path to model .cfg file'},
    'double': {'help': 'If specified, DDQN will be used', 'action': 'store_true'},
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
