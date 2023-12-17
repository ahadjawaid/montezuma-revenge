import torch
import torch.nn as nn
from typing import Callable

class ConvNet(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),  # Output: [16, 210, 160]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # Output: [16, 105, 80]
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), # Output: [32, 105, 80]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # Output: [32, 52, 40]
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), # Output: [64, 52, 40]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                   # Output: [64, 26, 20]
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),                       # Flatten the output for the fully connected layer
            nn.Linear(64 * 26 * 20, 1024),      # First fully connected layer
            nn.ReLU(),
            nn.Linear(1024, out_dim)            # Output layer with out_dim features
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class FeedForwardNN(nn.Module):
    def __init__(
            self, 
            in_dim: int, 
            hidden_dim: int, 
            out_dim: int, 
            n_layers: int, 
            Activation: Callable = None,
            Norm: Callable = None,
        ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            linear(in_dim, hidden_dim, Activation, Norm),
            *[linear(hidden_dim, hidden_dim, Activation, Norm) for _ in range(n_layers - 2)],
            linear(hidden_dim, out_dim, None)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.layers(x)

        return logits

def linear(in_dim: int, out_dim: int, Activation: Callable = None, Norm: Callable = None) -> nn.Module:
    layers = [nn.Linear(in_dim, out_dim)]

    if Activation is not None:
        layers.append(Activation())

    if Norm is not None:
        layers.append(Norm(out_dim))    

    return nn.Sequential(*layers)

def conv2d(in_dim: int, out_dim: int, ks=3, Activation: Callable = None, Pooling: Callable = None, **kwargs) -> nn.Module:
    layers = [nn.Conv2d(in_dim, out_dim, kernel_size=3, **kwargs)]

    if Activation is not None:
        layers.append(Activation())

    if Pooling is not None:
        layers.append(Pooling())

    return nn.Sequential(*layers)