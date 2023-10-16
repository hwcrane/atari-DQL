import torch
import gymnasium as gym
from agent import AtariAgent
from utils import convert_observation, wrap_env


def next_action(observation, policy_net, device):
    with torch.no_grad():
        return (
            policy_net(observation.to(device))
            .max(1)[1]
            .view(1, 1)
        )


def test(env, policy_net, num_episodes, video_folder, device):
    env = gym.wrappers.RecordVideo(env, video_folder=video_folder)
    for _ in range(num_episodes):
        # Reset the environment for a new episode and convert the initial observation
        observation, _ = env.reset()
        observation = convert_observation(observation)

        while True:
            # Choose the next action using a greedy policy (epsilon=0) since this is testing
            action = next_action(observation, policy_net, device)

            # Take the chosen action in the environment and receive the next observation and reward
            next_observation, reward, terminated, truncated, _ = env.step(
                action
            )

            next_observation = convert_observation(next_observation)

            # Update the current observation
            observation = next_observation

            done = truncated or terminated
            if done:
                break
    env.close()


