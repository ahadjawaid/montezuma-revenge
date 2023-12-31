from modules import FeedForwardNN, ConvNet
import torch.nn as nn
from typing import Tuple, Callable
import torch

class DQN(nn.Module):
    def __init__(
            self, 
            in_size: Tuple[int, int],
            conv_feats: int,
            hidden_dim: int, 
            out_dim: int, 
            n_layers: int, 
            Activation: Callable,
            Norm: Callable = None,
        ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            ConvNet(in_size, conv_feats),
            FeedForwardNN(conv_feats, hidden_dim, out_dim, n_layers, Activation, Norm)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.layers(x)

        return logits