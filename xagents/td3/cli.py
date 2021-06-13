cli_args = {
    'actor-model': {'help': 'Path to actor model .cfg file'},
    'critic-model': {'help': 'Path to critic model .cfg file'},
    'policy-delay': {
        'help': 'Delay after which, actor weights and target models will be updated',
        'type': int,
        'default': 2,
    },
    'gradient-steps': {'help': 'Number of iterations per train step', 'type': int},
    'tau': {
        'help': 'Value used for syncing target model weights',
        'type': float,
        'default': 0.005,
    },
    'policy-noise-coef': {
        'help': 'Coefficient multiplied by noise added to target actions',
        'type': float,
        'default': 0.2,
    },
    'noise-clip': {
        'help': 'Target noise clipping value',
        'type': float,
        'default': 0.5,
    },
}
