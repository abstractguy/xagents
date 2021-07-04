<p>
  <a href="https://github.com/emadboctorx/xagents/">
  </a>

  <h3 align="left">xagents - reusable, scalable, 
  performant reinforcement learning algorithms in tf2</h3>
  </p>

* [Installation](#installation)
* [Description](#description)
* [Features](#features)
* [Usage](#usage)
* [Command line options](#command-line-options)
* [Algorithms](#algorithms)
  * [A2C](#a2c)
  * [ACER](#acer)
  * [DDPG](#ddpg)
  * [DQN / DDQN](#dqn)
  * [PPO](#ppo)
  * [TD3](#td3)
  * [TRPO](#trpo)
* [License](#license)
* [Show your support](#show-your-support)
* [Contact](#contact)

### **Installation**
___

```sh
git clone https://github.com/emadboctorx/xagents
cd xagents
pip install .
```

**Notes:** 
* To be able to run atari environments, follow the instructions in [atari-py](https://github.com/openai/atari-py#roms)
to install [ROMS](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html)
  
* To be able to run the tests remotely, [pytest-xvfb](https://pypi.org/project/pytest-xvfb/) plugin will
be automatically installed but will require an additional step:
  * For macOS users, you'll have to install [Xquartz](https://www.xquartz.org)
    ```shell 
    brew install xquartz
    ```
  * For linux, you'll have to install [xvfb](https://www.x.org/releases/X11R7.6/doc/man/man1/Xvfb.1.xhtml)
    ```shell
    sudo apt-get install -y xvfb
    ```
    
3. **Verify installation**

```sh
xagents
```

**OUT:**

xagents 1.0

    Usage:
        xagents <command> <agent> [options] [args]
    
    Available commands:
        train      Train given an agent and environment
        play       Play a game given a trained agent and environment

<!-- DESCRIPTION -->
## **Description**
___
xagents is a tensorflow based mini-library which facilitates experimentation with
existing reinforcement learning algorithms, as well as the implementation of new agents by 
providing well tested components that can be easily modified or extended. It has
a selection of powerful algorithms that are ready to use directly or through command line.

<!-- FEATURES -->
## **Features**

* Tensorflow 2, highly performant agents.
* wandb monitoring.
* Multiple environments (All agents)
* Agents available through import or using the command line.
* Early stopping, and plateau learning rate throttling.
* Checkpoint and save trained models to .tf format upon improvement with resume
training available.
  
* Discrete and continuous environment spaces.
* Well tested components which should facilitate implementing
new agents through extending 
  [OnPolicy](https://github.com/emadboctorx/xagents/blob/5413b5a6e97347535c5067062254cc5d58405077/xagents/base.py#L552) 
  and 
  [OffPolicy](https://github.com/emadboctorx/xagents/blob/5413b5a6e97347535c5067062254cc5d58405077/xagents/base.py#L569) agents.
  
* All agents accept external keras models which are either passed
directly to agents in code or loaded from .cfg files passed through
  command line. Off-policy agents accept replay buffers as well.
  
* Training history can be saved to .parquet files for benchmarking.
* Visualization of training history of single / multiple agents.
* Random seeds to allow reproduction of results.
* Game play rendering which can be saved to .jpg frames or .mp4 vid.

## **Usage**

All agents are available through the command line:

    xagents <command> <agent> [options] [args]

Example:

    xagents train a2c --env PongNoFrameskip-v4 --target-reward 19 --preprocess

Or through direct importing:

    from xagents import A2C
    from xagents.utils.common import ModelReader, create_gym_env
    
    envs = create_gym_env('PongNoFrameskip-v4')
    model = ModelReader('/path/to/model.cfg', output_units=[6, 1], optimizer='adam').build_model()
    agent = A2C(envs, model)
    
Then either `max_steps` / `--max-steps` or `target_reward` / `--target-reward`
should be specified to start training:
    
    agent.fit(target_reward=19)

<!-- COMMAND LINE OPTIONS -->
## **Command line options**

Not all the flags listed below are available at once, and to know which 
ones are available to the command you passed you can use:

    xagents <command>

or

    xagents <command> <agent>

Which should list the exact commands that are available to the command + agent

Flags (Available for all agents)

| flags                         | help                                                                         | required   | default   |
|:------------------------------|:-----------------------------------------------------------------------------|:-----------|:----------|
| --env                         | gym environment id                                                           | True       | -         |
| --n-envs                      | Number of environments to create                                             | -          | 1         |
| --preprocess                  | If specified, states will be treated as atari frames                         | -          | -         |
|                               | and preprocessed accordingly                                                 |            |           |
| --no-scale                    | If specified, frames will not be scaled / normalized (divided by 255)        | -          | -         |
| --lr                          | Learning rate passed to a tensorflow.keras.optimizers.Optimizer              | -          | 0.0007    |
| --opt-epsilon                 | Epsilon passed to a tensorflow.keras.optimizers.Optimizer                    | -          | 1e-07     |
| --beta1                       | Beta1 passed to a tensorflow.keras.optimizers.Optimizer                      | -          | 0.9       |
| --beta2                       | Beta2 passed to a tensorflow.keras.optimizers.Optimizer                      | -          | 0.999     |
| --weights                     | Path(s) to model(s) weight(s) to be loaded by agent output_models            | -          | -         |
| --max-frame                   | If specified, max & skip will be applied during preprocessing                | -          | -         |
| --reward-buffer-size          | Size of the total reward buffer, used for calculating                        | -          | 100       |
|                               | mean reward value to be displayed.                                           |            |           |
| --gamma                       | Discount factor                                                              | -          | 0.99      |
| --display-precision           | Number of decimals to be displayed                                           | -          | 2         |
| --seed                        | Random seed                                                                  | -          | -         |
| --scale-factor                | Input scale divisor                                                          | -          | -         |
| --log-frequency               | Log progress every n games                                                   | -          | -         |
| --checkpoints                 | Path(s) to new model(s) to which checkpoint(s) will be saved during training | -          | -         |
| --history-checkpoint          | Path to .parquet file to save training history                               | -          | -         |
| --plateau-reduce-factor       | Factor multiplied by current learning rate when there is a plateau           | -          | 0.9       |
| --plateau-reduce-patience     | Minimum non-improvements to reduce lr                                        | -          | 10        |
| --early-stop-patience         | Minimum plateau reduces to stop training                                     | -          | 3         |
| --divergence-monitoring-steps | Steps after which, plateau and early stopping are active                     | -          | 500000    |
| --target-reward               | Target reward when reached, training is stopped                              | -          | -         |
| --max-steps                   | Maximum number of environment steps, when reached, training is stopped       | -          | -         |
| --monitor-session             | Wandb session name                                                           | -          | -         |

<!-- ALGORITHMS -->
## **Algorithms**

### *A2C*

### *ACER*

### *DDPG*

### *DQN-DDQN*

### *PPO*

### *TD3*

### *TRPO*


## **License**
___

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

## **Show your support**
___

Give a ⭐️ if this project helped you!

## **Contact**
___

Emad Boctor - emad_1989@hotmail.com

Project link: https://github.com/emadboctorx/xagents