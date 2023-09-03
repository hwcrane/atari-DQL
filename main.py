import torch
import gymnasium as gym
from agent import AtariAgent
import numpy as np
import time
import datetime
from torch.utils.tensorboard import SummaryWriter


def convert_observation(observation):
    """Converts the observation from a numpy array bounded 0-255 to torch Tensor 0-1"""
    return torch.from_numpy(np.array(observation).astype(np.float32) / 255)


def train(env: gym.Env, agent: AtariAgent, n_episodes: int, batch_size: int):
    writer = SummaryWriter()
    start_time = time.time()
    for episode in range(n_episodes):
        observation, _ = env.reset()
        observation = convert_observation(observation)

        total_reward = 0.0

        while True:
            action = agent.next_action(observation)

            next_observation, reward, terminated, truncated, info = env.step(
                action
            )

            done = truncated or terminated
            next_observation = convert_observation(next_observation)

            total_reward += reward   # type: ignore
            reward = torch.tensor([reward])

            agent.new_transition(observation, action, reward, next_observation, done)

            observation = next_observation

            agent.optimise(batch_size)

            if done:
                break

        writer.add_scalar('Reward', total_reward, episode)
        writer.add_scalar('Total Steps', agent.steps_done, episode)
        writer.add_scalar('Epsilon', agent.epsilon(), episode)
        writer.flush()
        # if episode % 20 == 0:
        #     print(
        #         f'{datetime.timedelta(seconds=int(time.time() - start_time))} - Episode: {episode}, Total Steps: {agent.steps_done},  Total Reward: {total_reward}, Epsilon: {agent.epsilon()}'
        #     )
    writer.close()


def test(env: gym.Env, agent: AtariAgent, n_episodes: int):
    env = gym.wrappers.RecordVideo(env, './videos/' + 'dqn_pong_video')
    for _ in range(n_episodes):
        observation, _ = env.reset()
        observation = convert_observation(observation)

        total_reward = 0.0

        while True:
            action = agent.next_action(observation, epsilon=0)

            next_observation, reward, terminated, truncated, info = env.step(
                action
            )
            next_observation = convert_observation(next_observation)
            total_reward += reward   # type: ignore
            reward = torch.tensor([reward])
            observation = next_observation

            done = truncated or terminated
            if done:
                break

    env.close()
    return

def wrap_env(env: gym.Env):
    env = gym.wrappers.GrayScaleObservation(env) # Make Greyscale
    env = gym.wrappers.ResizeObservation(env, (84, 84)) # Resize
    env = gym.wrappers.FrameStack(env, num_stack=4) # Stack Frames
    return env

if __name__ == '__main__':
    device = torch.device('mps' if torch.has_mps else 'cpu')

    # Hyperparameters
    BATCH_SIZE = 32

    env = gym.make('ALE/Pong-v5')
    env = wrap_env(env)
    sample = env.observation_space.sample()

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
        network_file = "PongModel2"
    )
    agent.steps_done = int(1.484e+6)

    train(env, agent, 500, BATCH_SIZE)
    torch.save(agent.policy_net, 'PongModel3')
    # test(env, agent, 1)
