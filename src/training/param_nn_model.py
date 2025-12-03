# src/training/params_nn_model.py

import torch
import torch.nn as nn

# We now use 23 features in encode_sample(...)
INPUT_DIM = 23
OUTPUT_DIM = 6   # delta, alpha, k_a, k_m, L, c


class ParamsNN(nn.Module):
    """
    Global neural net mapping full (gamme + vehicle) features -> 6 parametric coefficients.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, INPUT_DIM]
        returns: [B, OUTPUT_DIM]
        """
        return self.net(x)
