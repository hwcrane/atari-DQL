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
        network_file=None,
    ) -> None:
        """
        Creates a new Atari agent.

        Args:
            device (torch.device): The device (CPU or GPU) on which to run the agent.
            n_actions (int): The number of possible actions the agent can take.
            lr (float): The learning rate for the agent's neural network optimizer.
            epsilon_start (float): The initial exploration rate for epsilon-greedy policy.
            epsilon_end (float): The final exploration rate for epsilon-greedy policy.
            epsilon_decay (float): The rate at which epsilon decays over time.
            total_memory (int): The maximum capacity of the replay memory.
            initial_memory (int): The minimum number of transitions required in the replay memory before learning starts.
            gamma (float): The discount factor for future rewards in the Q-learning update.
            target_update (int): The frequency at which to update the target network.

            network_file (str, optional): A file to load pre-trained network weights from.
        """
        self.n_actions = n_actions
        self.device = device

        # Deep Q networkd for policy and target, with optimiser for policy
        if network_file:
            self.policy_net = torch.load(network_file).to(self.device)
            self.target_net = torch.load(network_file).to(self.device)
        else:
            # Create new policy and target networks if no pre-trained weights are provided
            self.policy_net = DQN(self.n_actions).to(device)
            self.target_net = DQN(self.n_actions).to(device)
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Optimiser for the policy network
        self.optimiser = Adam(self.policy_net.parameters(), lr=lr)

        self.steps_done = 0
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.target_update = target_update

        # Initialize the replay memory
        self.memory = ReplayMemory(capacity=total_memory)
        self.initial_memory = initial_memory

    def new_transition(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_observation: torch.Tensor,
        done: bool,
    ):
        """
        Stores a new transition in the replay memory.

        Args:
            observation (torch.Tensor): The current observation/state.
            action (torch.Tensor): The action taken.
            reward (torch.Tensor): The reward received.
            next_observation (torch.Tensor): The next observation/state.
            done (bool): A flag indicating if the episode is done.
        """

        self.memory.push(
            observation.to('cpu'),
            action.to('cpu'),
            reward.to('cpu'),
            next_observation.to('cpu') if not done else None,
        )

    def epsilon(self):
        """
        Calculates the current epsilon threshold for epsilon-greedy policy.

        Returns:
            float: The current epsilon value.
        """
        return self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * (1 - min(self.steps_done / self.epsilon_decay, 1))

    def next_action(
        self, observation: torch.Tensor, epsilon: float | None = None
    ) -> torch.Tensor:
        """
        Calculates the next action to be taken using an epsilon-greedy policy.

        Args:
            observation (torch.Tensor): The current observation/state.
            epsilon (float, optional): The epsilon value to use for epsilon-greedy policy.

        Returns:
            torch.Tensor: The chosen action.
        """
        if epsilon is None:
            epsilon = self.epsilon()

        self.steps_done += 1

        if random.random() > epsilon:
            with torch.no_grad():
                return (
                    self.policy_net(observation.to(self.device))
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
        """
        Performs one optimization step of the Q-network.

        Args:
            batch_size (int): The number of transitions to sample and use for optimization.
        """

        # Only start optimizing once there are enough transitions in memory
        if (
            len(self.memory) < self.initial_memory
            or len(self.memory) < batch_size
        ):
            return

        # Sample a batch of transitions from the replay memory
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        # Create a mask for non-final states in the batch
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )

        # Extract non-final next states and convert them to a tensor
        non_final_next_states = torch.stack(
            [s for s in batch.next_state if s is not None]
        ).to(self.device)

        # Convert states, actions, and rewards in the batch to tensors
        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        # Compute the predicted Q-values for the state-action pairs in the batch
        state_action_values = self.policy_net(state_batch).gather(
            1, action_batch
        )

        # Initialise the next state values as zeros
        next_state_values = torch.zeros(batch_size, device=self.device)
        # Update the next state values with values from the target network for non-final states
        next_state_values[non_final_mask] = (
            self.target_net(non_final_next_states).max(1)[0].detach()
        )

        # Compute the expected state-action values using the Bellman equation
        expected_state_action_values = (
            next_state_values * self.gamma
        ) + reward_batch

        # Compute the loss using the Huber loss (smooth L1 loss)
        loss = torch.nn.functional.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        # Zero the gradients of the policy network's parameters
        self.optimiser.zero_grad()

        # Compute the gradients and perform gradient clipping
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        # Update the policy network's parameters
        self.optimiser.step()

        # Update the target network's parameters if the target update interval is reached
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
