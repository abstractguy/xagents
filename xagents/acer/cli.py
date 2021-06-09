cli_args = {
    'model': {'help': 'Path to model .cfg file'},
    'ema-alpha': {
        'help': 'Moving average decay passed to tf.train.ExponentialMovingAverage()',
        'type': float,
        'default': 0.99,
    },
    'replay-ratio': {
        'help': 'Lam value passed to np.random.poisson()',
        'type': int,
        'default': 4,
    },
    'epsilon': {
        'help': 'epsilon used in gradient updates',
        'type': float,
        'default': 1e-6,
    },
    'importance-c': {
        'help': 'Importance weight truncation parameter.',
        'type': float,
        'default': 10.0,
    },
    'delta': {
        'help': 'delta param used for trust region update',
        'type': float,
        'default': 1,
    },
    'trust-region': {
        'help': 'True by default, if this flag is specified,\n'
        'no trust region updates will not be used',
        'action': 'store_false',
    },
}
