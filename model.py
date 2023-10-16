import torch.nn as nn
import torch

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
class DQN(nn.Module):
    """Deep Q Network"""

    def __init__(self, n_actions: int) -> None:
        super().__init__()

        # Convolutional layers of the network
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Linear layers of the network
        self.linear_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(), nn.Linear(512, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize the input by scaling it to [0, 1]
        x = x.float() / 255

        # Pass input through the convolutional layers
        x = self.conv_layers(x)

        # Flatten the output from the convolutional layers
        x = x.view(-1, 64 * 7 * 7)

        # Pass the flattened output through the linear layers
        x = self.linear_layers(x)
        return x
