import torch
import gymnasium as gym
from agent import AtariAgent
from utils import convert_observation, wrap_env

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
        network_file='PongModel',
    )

    test(env, agent, 1)
