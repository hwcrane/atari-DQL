import argparse
import configparser

from test import test


def load_parameters_from_config(config_file, mode="Training"):
    config = configparser.ConfigParser()
    config.read(config_file)
    if mode not in config:
        print(f"Error: '{mode}' section not found in the configuration file '{config_file}'.")
        exit(1)
    return config[mode]


def testing(config_data):
    env_name = config_data.get('env')
    model_path = config_data.get('model_path')
    episodes = int(config_data.get('episodes'))
    video_folder = config_data.get('video_folder')

    import torch
    import gymnasium as gym
    from utils import wrap_env

    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.has_mps
        else 'cpu'
    )

    env = gym.make(env_name, render_mode='rgb_array')
    env = wrap_env(env)

    policy_net = torch.load(model_path).to(device)

    test(env, policy_net, episodes, video_folder, device)


def training(config_data):
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

    # Load expensive imports here to speed up program start time
    import torch
    import gymnasium as gym
    from agent import AtariAgent
    from train import train
    from utils import wrap_env

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['Training', 'Testing'], help='Training or Testing')
    parser.add_argument('--config', '-c', help='INI configuration file', required=True)

    args = parser.parse_args()
    config_path = args.config
    config_data = load_parameters_from_config(config_path, args.mode)

    if args.mode == 'Training':
        training(config_data)
    else:
        testing(config_data)


if __name__ == '__main__':
    main()
