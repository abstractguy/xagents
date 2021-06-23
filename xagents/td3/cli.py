from xagents import ddpg

ddpg_args = ddpg.cli_args
td3_args = {
    'policy-delay': {
        'help': 'Delay after which, actor weights and target models will be updated',
        'type': int,
        'default': 2,
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
cli_args = ddpg_args.copy()
cli_args.update(td3_args)
