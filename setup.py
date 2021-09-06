from setuptools import find_packages, setup

install_requires = [
    'tensorflow_probability==0.13.0',
    'opencv_python_headless==4.5.3.56',
    'tensorflow_addons==0.14.0',
    'numpy==1.19.5',
    'pandas==1.3.2',
    'gym[atari,box2d]==0.19.0',
    'tensorflow==2.6.0',
    'wandb==0.12.1',
    'tabulate==0.8.9',
    'pyglet==1.5.15',
    'pytest==6.2.5',
    'pyarrow==5.0.0',
    'fastparquet==0.7.1',
    'matplotlib==3.4.3',
    'optuna==2.9.1',
    'termcolor==1.1.0',
]

setup(
    name='xagents',
    version='1.0.1',
    packages=find_packages(),
    url='https://github.com/schissmantics/xagents',
    license='MIT',
    author='schissmantics',
    author_email='schissmantics@outlook.com',
    description='Implementations of deep reinforcement learning algorithms in tensorflow 2.5',
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
