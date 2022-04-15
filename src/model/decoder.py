from typing import Tuple

import numpy as np
import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, image_size: Tuple[int, int, int] = (1, 64, 64),
                 latent_dim: int = 1024,
                 n_features: int = 5):
        super(Decoder, self).__init__()

        # Layer parameters
        kernel_size = 4
        hidden_dim = 1024

        self.latent_dim = latent_dim
        self.image_size = image_size
        self.n_features = n_features
        self.in_size = self.latent_dim * self.n_features
        self.reshape = (128, kernel_size, kernel_size)

        n_channels = self.image_size[0]

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, np.product(self.reshape))

        # Convolutional layers
        cnn_kwargs = dict(kernel_size=kernel_size, stride=2, padding=1)

        # self.convT5 = nn.ConvTranspose2d(256, 128, **cnn_kwargs)
        self.convT4 = nn.ConvTranspose2d(128, 64, **cnn_kwargs)
        self.convT3 = nn.ConvTranspose2d(64, 32, **cnn_kwargs)
        self.convT2 = nn.ConvTranspose2d(32, 16, **cnn_kwargs)
        self.convT1 = nn.ConvTranspose2d(16, n_channels, **cnn_kwargs)

        self.activation = torch.nn.GELU()
        self.final_activation = torch.nn.Sigmoid()

    def forward(self, z):
        # Fully connected layers with ReLu activations
        x = self.activation(self.lin1(z))
        x = self.activation(self.lin3(x))

        x = x.view(-1, *self.reshape)

        # Convolutional layers with ReLu activations
        # x = self.activation(self.convT5(x))
        x = self.activation(self.convT4(x))
        x = self.activation(self.convT3(x))
        x = self.activation(self.convT2(x))

        # Sigmoid activation for final conv layer
        x = self.final_activation(self.convT1(x))

        return x
