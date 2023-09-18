# Atari Reinforcement Learning Agent

![Trained Agent Video](videos/dqn_pong_video/rl-video-episode-0.mp4)

This repository contains code for training and testing an Atari reinforcement learning agent using the Deep Q-Network (DQN) algorithm. The agent is capable of learning to play various Atari games provided by the OpenAI Gymnasium environment.

## Overview

The Atari reinforcement learning agent is implemented using PyTorch and Gymnasium, and it follows the DQN algorithm. The agent learns to play Atari games by interacting with the environment, storing experiences in a replay memory, and optimizing its Q-network.

The codebase includes the following components:

- `dqn.py`: Definition of the Deep Q-Network (DQN) model.
- `replay_memory.py`: Implementation of the replay memory for storing and sampling experiences.
- `agent.py`: The main Atari agent class that encapsulates the training and testing logic.
- `train.py`: Script for training the agent.
- `test.py`: Script for testing the trained agent.
- `utils.py`: Utility functions for preprocessing observations and other tasks.
- `videos/`: A directory where recorded videos of the trained agent's gameplay are saved.


