# src/parametric/param_model_utils.py

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


PARAM_MODEL_FEATURES: List[str] = [
    "make",
    "model",
    "year",
    "trim",
    # later you can add: body_style, fuel_type, drivetrain, transmission, segment, etc.
]


class ParamNet(nn.Module):
    """
    Neural network for NNmarket:
      input: encoded Car Market features (one vector for {make, model, year, trim, ...})
      output: 7 unconstrained parameters (V0, delta, alpha, k_a, k_m, L, c)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        theta_unconstrained = self.fc_out(h)
        return theta_unconstrained


def unconstrained_to_params_torch(theta: torch.Tensor):
    """
    torch version of unconstrained -> constrained mapping.
    theta: (batch, 7)
    Returns a dict of constrained tensors.
    """

    def softplus(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x)

    def sigmoid(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    t_V0, t_delta, t_alpha, t_k_a, t_k_m, t_L, t_c = theta.unbind(dim=-1)

    V0 = softplus(t_V0) + 1e-3
    delta = sigmoid(t_delta)
    alpha = softplus(t_alpha)
    k_a = softplus(t_k_a)
    k_m = softplus(t_k_m)
    L = softplus(t_L)
    c = softplus(t_c)

    return {
        "V0": V0,
        "delta": delta,
        "alpha": alpha,
        "k_a": k_a,
        "k_m": k_m,
        "L": L,
        "c": c,
    }
