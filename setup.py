from setuptools import find_packages, setup

install_requires = [
    'gym[atari, box2d] == 0.21.0',
    'matplotlib == 3.4.3',
    'numpy == 1.19.5',
    'opencv - python - headless == 4.5.4.58',
    'optuna == 2.10.0',
    'pandas == 1.3.4',
    'pyarrow == 6.0.1',
    'pytest == 6.2.5',
    'setuptools == 58.5.3',
    'tensorflow == 2.7.0',
    'tensorflow - probability == 0.15.0',
    'termcolor == 1.1.0',
    'wandb == 0.12.7',
    'tensorflow_addons == 0.15.0',
    'tabulate == 0.8.9',
    'pyglet == 1.5.15',
    'fastparquet == 0.7.2',
]

setup(
    name='xagents',
    version='1.0.1',
    packages=find_packages(),
    url='https://github.com/schissmantics/xagents',
    license='MIT',
    author='schissmantics',
    author_email='schissmantics@outlook.com',
    description='Implementations of deep reinforcement learning algorithms in tensorflow 2.7',
    include_package_data=True,
    setup_requires=['numpy==1.19.5'],
    install_requires=install_requires,
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'xagents=xagents.cli:execute',
        ],
    },
)
