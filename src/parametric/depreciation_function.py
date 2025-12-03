# src/parametric/depreciation_function.py

# Target formula:
#   V(t, m) = L + (V0 − L) * (1 − δ (1 − e^(−α t)))
#             * exp(−k_a * t) * exp(−k_m * m) * exp(c * t)
#
# i.e.
#   V(t, m) = L + (V0 − L) * (1 − δ (1 − e^(−α t))) *
#             exp( (-k_a + c) * t - k_m * m )

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

import torch

Number = Union[float, int]


# ---------------------------------------------------------------------------
# 1. Dataclass for scalar usage
# ---------------------------------------------------------------------------

@dataclass
class DepreciationParams:
    """
    Container for the six learned parameters of a gamme.

    - delta: time decay speed          (in the (1 - δ(1 - e^-αt)) term)
    - alpha: time decay curvature      (inside e^-αt)
    - k_a: age effect                  (exp(-k_a * t))
    - k_m: mileage sensitivity         (exp(-k_m * m))
    - L: floor value                   (V asymptotes around >= L)
    - c: appreciation factor           (exp(c * t))
    """
    delta: float
    alpha: float
    k_a: float
    k_m: float
    L: float
    c: float


# ---------------------------------------------------------------------------
# 2. Pure Python / scalar implementation (for notebooks, quick checks)
# ---------------------------------------------------------------------------

def scalar_price_from_params(
    V0: Number,
    age_years: Number,
    mileage: Number,
    params: DepreciationParams,
) -> float:
    """
    Scalar (non-torch) version of the parametric price function.

    Implements:

        V(t, m) = L + (V0 − L) * (1 − δ (1 − e^(−α t))) *
                  exp(−k_a * t) * exp(−k_m * m) * exp(c * t)

    where:
        t = age_years   (>= 0)
        m = mileage     (you choose units; km here)

    Args:
        V0: initial value at t=0, from gammes.V0
        age_years: age of the vehicle in years (>= 0)
        mileage: mileage (e.g., km)
        params: DepreciationParams(delta, alpha, k_a, k_m, L, c)

    Returns:
        Predicted price as a float.
    """
    delta = float(params.delta)
    alpha = float(params.alpha)
    k_a = float(params.k_a)
    k_m = float(params.k_m)
    L = float(params.L)
    c = float(params.c)

    V0 = float(V0)
    t = max(0.0, float(age_years))
    m = max(0.0, float(mileage))

    # Time shape term: (1 - δ (1 - e^-αt))
    time_shape = 1.0 - delta * (1.0 - math.exp(-alpha * t))

    # Exponential terms: exp(-k_a t) * exp(-k_m m) * exp(c t)
    expo = -k_a * t - k_m * m + c * t
    expo_term = math.exp(expo)

    price = L + (V0 - L) * time_shape * expo_term
    return float(price)


# ---------------------------------------------------------------------------
# 3. Torch implementation used during training (batches, gradients, etc.)
# ---------------------------------------------------------------------------

def price_from_params(
    V0: torch.Tensor,            # [B]
    age_years: torch.Tensor,     # [B]
    mileage: torch.Tensor,       # [B]
    params: torch.Tensor,        # [B, 6] = [delta, alpha, k_a, k_m, L, c]
) -> torch.Tensor:
    """
    Vectorized PyTorch version of the parametric price function.

    Implements:

        V(t, m) = L + (V0 − L) * (1 − δ (1 − e^(−α t))) *
                  exp(−k_a * t) * exp(−k_m * m) * exp(c * t)

    with extra clamping for numerical stability.
    """
    V0 = V0.float()
    t = age_years.float().clamp_min(0.0)
    m = mileage.float().clamp_min(0.0)
    params = params.float()

    delta, alpha, k_a, k_m, L, c = torch.unbind(params, dim=1)  # each [B]

    # ---- Clamp parameters to reasonable ranges to avoid explosions ----
    # You can tweak these later based on empirical distributions.
    delta = delta.clamp(0.0, 5.0)
    alpha = alpha.clamp(0.0, 5.0)
    k_a = k_a.clamp(-5.0, 5.0)
    k_m = k_m.clamp(0.0, 5.0)
    c = c.clamp(-1.0, 1.0)

    # Time shape term: (1 - δ (1 - e^-αt))
    time_shape = 1.0 - delta * (1.0 - torch.exp(-alpha * t))

    # Exponential terms: exp(−k_a * t) * exp(−k_m * m) * exp(c * t)
    expo = -k_a * t - k_m * m + c * t

    # Clamp exponent to avoid overflow in exp()
    expo = expo.clamp(min=-50.0, max=50.0)
    expo_term = torch.exp(expo)

    price = L + (V0 - L) * time_shape * expo_term

    # Final safety: replace NaN/inf with 0 (or some fallback)
    price = torch.where(torch.isfinite(price), price, torch.zeros_like(price))

    return price
