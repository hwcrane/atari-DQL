from collections import deque

import torch
import gymnasium as gym
from agent import AtariAgent
from torch.utils.tensorboard import SummaryWriter
from utils import convert_observation


def train(env: gym.Env, agent: AtariAgent, n_episodes: int, batch_size: int):
    """
    Train the Atari agent using the specified environment.

    Args:
        env (gym.Env): The Gym environment used for training.
        agent (AtariAgent): The AtariAgent that will be trained.
        n_episodes (int): The number of episodes to train the agent.
        batch_size (int): The batch size used for Q-network optimization.
    """

    rewards = deque(maxlen=100)
    writer = SummaryWriter()
    for episode in range(n_episodes):
        # Reset the environment for a new episode and convert the initial observation

        observation, _ = env.reset()
        observation = convert_observation(observation)

        total_reward = 0.0

        while True:
            # Choose the next action using the agent's epsilon-greedy policy
            action = agent.next_action(observation)

            # Take the chosen action in the environment and receive the next observation and reward
            next_observation, reward, terminated, truncated, info = env.step(
                action
            )

            done = truncated or terminated
            next_observation = convert_observation(next_observation)

            total_reward += reward  # type: ignore
            rewards.append(reward)
            reward = torch.tensor([reward])

            # Store the transition in the replay memory
            agent.new_transition(
                observation, action, reward, next_observation, done
            )

            # Update the current observation
            observation = next_observation

            # Perform one optimization step of the Q-network
            agent.optimise(batch_size)

            if done:
                break

        # Log episode-related information to TensorBoard
        writer.add_scalar('Reward', total_reward, episode)
        writer.add_scalar('Epsilon', agent.epsilon(), episode)
        writer.flush()
    writer.close()
    env.close()
