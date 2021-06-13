from xagents import a2c, ppo

a2c_args = a2c.cli_args
ppo_args = ppo.cli_args
trpo_args = {
    'actor-model': {'help': 'Path to actor model .cfg file'},
    'critic-model': {'help': 'Path to critic model .cfg file'},
    'max-kl': {
        'help': 'Maximum KL divergence used for calculating Lagrange multiplier',
        'type': float,
        'default': 1e-3,
    },
    'cg-iterations': {
        'help': 'Gradient conjugation iterations per train step',
        'type': int,
        'default': 10,
    },
    'cg-residual-tolerance': {
        'help': 'Gradient conjugation residual tolerance parameter',
        'type': float,
        'default': 1e-10,
    },
    'cg-damping': {
        'help': 'Gradient conjugation damping parameter',
        'type': float,
        'default': 1e-3,
    },
    'actor-iterations': {
        'help': 'Actor optimization iterations per train step',
        'type': int,
        'default': 10,
    },
    'critic-iterations': {
        'help': 'Critic optimization iterations per train step',
        'type': int,
        'default': 3,
    },
    'fvp-n-steps': {
        'help': 'Value used to skip every n-frames used to calculate FVP',
        'type': int,
        'default': 5,
    },
    'entropy-coef': {
        'help': 'Entropy coefficient for loss calculation',
        'type': float,
        'default': 0,
    },
    'lam': {
        'help': 'GAE-Lambda for advantage estimation',
        'type': float,
        'default': 1.0,
    },
    'n-steps': {'help': 'Transition steps', 'type': int, 'default': 512},
}
cli_args = a2c_args.copy()
cli_args.update(ppo_args)
cli_args.update(trpo_args)
del cli_args['model']
