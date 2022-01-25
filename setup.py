from setuptools import find_packages, setup

install_requires = open('requirements.txt').read().splitlines()

setup(
    name='xagents',
    version='1.0.1',
    packages=find_packages(),
    url='https://github.com/alternativebug/xagents',
    license='MIT',
    author='alternativebug',
    author_email='alternativebug@outlook.com',
    description='Implementations of deep reinforcement learning algorithms in tensorflow 2.7',
    include_package_data=True,
    install_requires=install_requires,
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'xagents=xagents.cli:execute',
        ],
    },
)
