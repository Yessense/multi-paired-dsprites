from typing import Tuple

import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, latent_dim: int = 1024):
        super(MLP, self).__init__()

        self.in_lin = nn.Linear(latent_dim * 2, latent_dim * 2)
        self.hidden_lin = nn.Linear(latent_dim * 2, latent_dim)
        self.out_lin = nn.Linear(latent_dim, latent_dim)

        self.activation = nn.GELU()

    def forward(self, x):
        x = self.activation(self.in_lin(x))
        x = self.activation(self.hidden_lin(x))
        x = self.activation(self.out_lin(x))
        return x
