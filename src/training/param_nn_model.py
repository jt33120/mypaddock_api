# src/training/params_nn_model.py
from __future__ import annotations

import torch
import torch.nn as nn

INPUT_DIM = 23
OUTPUT_DIM = 6  # delta, alpha, k_a, k_m, L, c


class HierarchicalParamsNN(nn.Module):
    """
    Hierarchical model:
      params = theta_gamme + residual_scale * delta_params(x)

    - theta_gamme is a learnable per-gamme base vector (embedding table)
    - delta_params(x) comes from an MLP over features
    """

    def __init__(
        self,
        num_gammes: int,
        hidden: int = 64,
        dropout: float = 0.2,
        residual_scale: float = 0.25,
    ):
        super().__init__()
        self.residual_scale = float(residual_scale)

        # Base parameters per gamme
        self.gamme_base = nn.Embedding(num_gammes, OUTPUT_DIM)
        nn.init.normal_(self.gamme_base.weight, mean=0.0, std=0.02)

        # Residual network (feature -> delta params)
        self.residual_net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor, gamme_idx: torch.Tensor) -> torch.Tensor:
        """
        x: [B, INPUT_DIM]
        gamme_idx: [B] long
        returns params: [B, OUTPUT_DIM]
        """
        base = self.gamme_base(gamme_idx)              # [B, 6]
        delta = self.residual_net(x)                   # [B, 6]
        return base + (self.residual_scale * delta)
