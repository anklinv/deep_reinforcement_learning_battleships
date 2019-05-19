# Playing Battleships with Deep Reinforcement Learning
Solving the classical board game [battleships](https://en.wikipedia.org/wiki/Battleship_(game)) using a deep reinforcement agent. The goal is to minimize the number of steps to complete the game.

## Results
The best agent manages to achieve a mean game length of 61. Training took around 50 hours on a CPU, where the agent played a total of 3.3 million games.

![plot](https://github.com/anklinv/deep_reinforcement_learning_battleships/blob/master/plots/mean_with_25_and_75_percentiles.svg)

Below you can see an animation of the agent performing the task as well as the action probabilities.

![animation](https://github.com/anklinv/deep_reinforcement_learning_battleships/blob/master/animations/battleships_61.gif)

Other attempts include DQN which did not learn anything. I suspect that the reason for this is the highly stochastic nature of the game prohibits the agent to learn a good Q function.

## Installation
Clone repository
```Bash
git clone https://github.com/anklinv/deep_reinforcement_learning_battleships
```

Install dependencies
```Bash
cd deep_reinforcement_learning_battleships
pip install -e .
pip install -e ./coding_challenge/
```

## Training
To load the best model and save an animation of a game
```Bash
python policy_gradient_agent.py
```

To train from scratch
```Bash
python policy_gradient_agent.py --train
```

To load the latest checkpoint (located in folder models) and continue Training
```Bash
python policy_gradient_agent.py --train --load
```

To plot all the timesteps
```Bash
python policy_gradient_agent.py --plot
```

## Folder structure
- __animations__: contains all animation of games performed using the best model
- __coding_challenge__: contains the game environment
- __data__: contains a matrix with all game lengths during the experiment
- __dqn_agent__: failed attempt of using DQN to learn the game
- __models__: contains saved models
- __plots__: contains all plots
