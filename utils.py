import torch
import numpy as np
import gymnasium as gym

def convert_observation(observation):
    """Converts the observation from a numpy array to a torch tensor"""
    return torch.from_numpy(np.array(observation))


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
