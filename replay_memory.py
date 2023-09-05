from collections import namedtuple, deque
import random

# Defines a named tuple 'Transition' to represent a single transition in the replay memory.
# A transition contains a state, an action, a reward, and the next state.
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))

class ReplayMemory:
    """
    Replay Memory for storing and sampling past experiences (transitions) for training reinforcement learning agents.
    
    Args:
        capacity (int): The maximum capacity of the replay memory.
    """
    def __init__(self, capacity: int) -> None:
        """
        Initializes the ReplayMemory with a specified capacity.
        
        Args:
            capacity (int): The maximum capacity of the replay memory.
        """
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """
        Adds a new transition to the replay memory.

        Args:
            *args: A tuple containing a state, an action, a reward, and the next state.
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """
        Randomly samples a batch of transitions from the replay memory.

        Args:
            batch_size (int): The number of transitions to sample in the batch.

        Returns:
            list: A list of sampled transitions.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Returns the current number of stored transitions in the replay memory.

        Returns:
            int: The number of stored transitions.
        """
        return len(self.memory)

