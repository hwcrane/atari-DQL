import torch
import gymnasium as gym
from agent import AtariAgent
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def convert_observation(observation):
    """Converts the observation from a numpy array to a torch tensor"""
    return torch.from_numpy(np.array(observation))


def train(env: gym.Env, agent: AtariAgent, n_episodes: int, batch_size: int):
    """
    Train the Atari agent using the specified environment.

    Args:
        env (gym.Env): The Gym environment used for training.
        agent (AtariAgent): The AtariAgent that will be trained.
        n_episodes (int): The number of episodes to train the agent.
        batch_size (int): The batch size used for Q-network optimization.
    """

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

            total_reward += reward   # type: ignore
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
        writer.add_scalar('Total Steps', agent.steps_done, episode)
        writer.add_scalar('Epsilon', agent.epsilon(), episode)
        writer.flush()
    writer.close()
    env.close()


def test(env: gym.Env, agent: AtariAgent, n_episodes: int):
    """
    Test the trained Atari agent in the specified environment.

    Args:
        env (gym.Env): The Gym environment used for testing.
        agent (AtariAgent): The trained AtariAgent to be tested.
        n_episodes (int): The number of episodes to run for testing.
    """

    # Wrap the environment to record a video of the testing sessions
    env = gym.wrappers.RecordVideo(env, './videos/' + 'dqn_pong_video')
    for _ in range(n_episodes):

        # Reset the environment for a new episode and convert the initial observation
        observation, _ = env.reset()
        observation = convert_observation(observation)

        total_reward = 0.0

        while True:
            # Choose the next action using a greedy policy (epsilon=0) since this is testing
            action = agent.next_action(observation, epsilon=0)

            # Take the chosen action in the environment and receive the next observation and reward
            next_observation, reward, terminated, truncated, _ = env.step(
                action
            )

            next_observation = convert_observation(next_observation)
            total_reward += reward   # type: ignore
            reward = torch.tensor([reward])

            # Update the current observation
            observation = next_observation

            done = truncated or terminated
            if done:
                break

    env.close()


def wrap_env(env: gym.Env):
    """
    Prepare a Gym environment for training or testing with an Atari agent.
    
    Args:
        env (gym.Env): The original Gym environment to be wrapped.
        
    Returns:
        gym.Env: The wrapped Gym environment.
    """

    # Convert observations to grayscale
    env = gym.wrappers.GrayScaleObservation(env)
    
    # Resize observations to a smaller resolution
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    
    # Stack multiple consecutive frames to provide temporal information to the agent
    env = gym.wrappers.FrameStack(env, num_stack=4)
    return env

if __name__ == '__main__':
    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.has_mps
        else 'cpu'
    )

    # Hyperparameters
    BATCH_SIZE = 32

    env = gym.make('ALE/Pong-v5', render_mode='rgb_array')
    env = wrap_env(env)

    agent = AtariAgent(
        device=device,
        n_actions=4,
        lr=1e-4,
        epsilon_start=1,
        epsilon_end=0.02,
        epsilon_decay=200_000,
        total_memory=100_000,
        initial_memory=10_000,
        gamma=0.99,
        target_update=1000,
        # network_file='PongModel4',
    )

    # train(env, agent, 1000, BATCH_SIZE)
    # torch.save(agent.policy_net, 'PongModel4')
    # test(env, agent, 1)
