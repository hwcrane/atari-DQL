from collections import deque
import gymnasium as gym
from torch.optim import Adam
from replay_memory import ReplayMemory, Transition
from model import DQN
import math
import torch
import random


class AtariAgent:
    def __init__(
        self,
        device: torch.device,
        n_actions: int,
        lr: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
        total_memory: int,
        initial_memory: int,
        gamma: float,
        target_update: int,
    ) -> None:
        """Creates a new Atari agent"""
        self.n_actions = n_actions
        self.device = device

        # Deep Q networkd for policy and target, with optimiser for policy
        self.policy_net = DQN(self.n_actions).to(device)
        self.target_net = DQN(self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimiser = Adam(self.policy_net.parameters(), lr=lr)

        self.steps_done = 0
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.target_update = target_update

        self.observation_buffer = deque(maxlen=4)
        self.memory = ReplayMemory(capacity=total_memory)
        self.initial_memory = initial_memory

    def reset_observation_buffer(self):
        """Clears the observation buffer, this shoulf be used every time the environment resets"""
        self.observation_buffer.clear()

    def new_transition(
        self,
        new_observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: bool,
    ):
        """Stores a new transition in the replay memory. If the observation buffer is not full, the transition is not stored"""
        # If there are not enough states in the state buffer to stack them, do not store new transition in memory
        if len(self.observation_buffer) < 4:
            self.observation_buffer.append(new_observation)
            return

        # Stack existing buffer
        stacked_state = torch.stack(list(self.observation_buffer))

        # Append new observation to buffer and then stack it
        self.observation_buffer.append(new_observation)
        next_stacked_state = torch.stack(list(self.observation_buffer))

        # Store transition
        self.memory.push(stacked_state.to("cpu"), action.to("cpu"), reward.to("cpu"), next_stacked_state.to("cpu"))

        if done:
            self.reset_observation_buffer()

    def epsilon(self):
        """Calculate's the current epsilon threshold"""
        return self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * math.exp(-1.0 * self.steps_done / self.epsilon_decay)

    def new_observation(self, observation: torch.Tensor):
        self.observation_buffer.append(observation)

    def next_action(self) -> torch.Tensor:
        """
        Calculates the next action to be taken,
        there is a 1-epsilon probability that a random action will be taken
        """
        epsilon = self.epsilon()

        self.steps_done += 1

        if random.random() > epsilon and len(self.memory) == 4:
            with torch.no_grad():
                return (
                    self.policy_net(
                        torch.stack(list(self.observation_buffer)).to(
                            self.device
                        )
                    )
                    .max(1)[1]
                    .view(1, 1)
                )
        else:
            return torch.tensor(
                [[random.randrange(self.n_actions)]],
                device=self.device,
                dtype=torch.long,
            )

    def optimise(self, batch_size: int):
        # Only start optimising once a sufficient memory has been created
        if (
            len(self.memory) < self.initial_memory
            or len(self.memory) < batch_size
        ):
            return

        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state_stack).to(self.device)
        next_state_batch = torch.stack(batch.next_state_stack).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        state_action_values = self.policy_net(state_batch).gather(
            1, action_batch
        )

        next_state_values = (
            self.target_net(next_state_batch).max(1)[0].detach()
        )

        expected_state_action_values = (
            next_state_values * self.gamma
        ) + reward_batch

        loss = torch.nn.functional.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        self.optimiser.zero_grad()

        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimiser.step()

        if self.steps_done % self.target_update:
            self.target_net.load_state_dict(self.policy_net.state_dict())
