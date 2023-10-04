import argparse
import configparser

import torch
import gymnasium as gym
from agent import AtariAgent
from train import train
from utils import wrap_env


def load_parameters_from_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    if 'Parameters' not in config:
        print(f"Error: 'Parameters' section not found in the configuration file '{config_file}'.")
        exit(1)
    return config['Parameters']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help='INI configuration file', required=True)

    config_path = parser.parse_args().config
    config_data = load_parameters_from_config(config_path)

    env_name = config_data.get('env')
    actions = int(config_data.get('actions'))
    eps_start = float(config_data.get('eps_start'))
    eps_end = float(config_data.get('eps_end'))
    eps_decay = float(config_data.get('eps_decay'))
    memory_size = int(config_data.get('memory_size'))
    learning_rate = float(config_data.get('learning_rate'))
    initial_memory = int(config_data.get('initial_memory'))
    gamma = float(config_data.get('gamma'))
    target_update = int(config_data.get('target_update'))
    batch_size = int(config_data.get('batch_size'))
    model_path = config_data.get('model_path')
    episodes = int(config_data.get('episodes'))

    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.has_mps
        else 'cpu'
    )

    env = gym.make(env_name)
    env = wrap_env(env)

    agent = AtariAgent(
        device=device,
        n_actions=actions,
        lr=learning_rate,
        epsilon_start=eps_start,
        epsilon_end=eps_end,
        epsilon_decay=eps_decay,
        total_memory=memory_size,
        initial_memory=initial_memory,
        gamma=gamma,
        target_update=target_update,
    )

    train(env, agent, batch_size, episodes)
    torch.save(agent.policy_net.to('cpu'), model_path)


if __name__ == '__main__':
    main()
