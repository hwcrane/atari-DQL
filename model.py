import torch.nn as nn
import torch

class DQN(nn.Module):
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
            nn.Linear(64 * 7 * 7, 512), 
            nn.ReLU(), 
            nn.Linear(512, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)

        # X is flattened before passing into linear layers
        x = x.view(x.size(0), -1)

        x = self.linear_layers(x)
        return x
