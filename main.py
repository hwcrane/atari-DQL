import torch
import gymnasium as gym
from agent import AtariAgent
from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np
import time
import datetime


def convert_observation(observation):
    """Converts the observation from 210x160 RGB numpy to 84x84 grayscale torch Tensor"""
    transformed_obs = rgb2gray(observation)
    transformed_obs = resize(transformed_obs, (84, 84), mode='constant')
    return torch.from_numpy(transformed_obs.astype(np.float32))


def train(env: gym.Env, agent: AtariAgent, n_episodes: int, batch_size: int):
    start_time = time.time()
    for episode in range(n_episodes):
        observation, _ = env.reset()
        observation = convert_observation(observation)
        agent.reset_observation_buffer()
        agent.new_observation(observation)

        total_reward = 0.0

        while True:
            action = agent.next_action()

            next_observation, reward, terminated, truncated, info = env.step(
                action
            )
            next_observation = convert_observation(next_observation)

            done = truncated or terminated

            total_reward += reward   # type: ignore
            reward = torch.tensor([reward])

            agent.new_transition(next_observation, action, reward, done)

            observation = next_observation

            if done:
                break

        if episode % 20 == 0:
            print(
                f'{datetime.timedelta(seconds=int(time.time() - start_time))} - Episode: {episode}, Total Steps: {agent.steps_done},  Total Reward: {total_reward}, Epsilon: {agent.epsilon()}'
            )


if __name__ == '__main__':
    device = torch.device('mps' if torch.has_mps else 'cpu')

    # Hyperparameters
    BATCH_SIZE = 32

    env = gym.make('PongNoFrameskip-v4')

    agent = AtariAgent(
        device=device,
        n_actions=env.action_space.n,
        lr=1e-4,
        epsilon_start=1,
        epsilon_end=0.02,
        epsilon_decay=1_000_000,
        total_memory=100_000,
        initial_memory=10_000,
        gamma=0.99,
        target_update=1000,
    )

    train(env, agent, 400, BATCH_SIZE)
    torch.save(agent.policy_net, "PongModel")
