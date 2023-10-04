import torch
import gymnasium as gym
from agent import AtariAgent
from torch.utils.tensorboard import SummaryWriter
from utils import convert_observation, wrap_env

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

if __name__ == '__main__':
    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.has_mps
        else 'cpu'
    )
    print(device)

    # Hyperparameters
    BATCH_SIZE = 32

    env = gym.make('ALE/Breakout-v5')
    env = wrap_env(env)

    agent = AtariAgent(
        device=device,
        n_actions=4,
        lr=1e-4,
        epsilon_start=1,
        epsilon_end=0.02,
        epsilon_decay=500_000,
        total_memory=100_000,
        initial_memory=10_000,
        gamma=0.99,
        target_update=10000,
    )

    train(env, agent, 5000, BATCH_SIZE)
    torch.save(agent.policy_net.to('cpu'), "BreakoutModel")
