from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class UniversalParams:
    """
    Constrained parameters for the universal depreciation model.

    V(t,m) = L + (V0 - L) * (1 - δ(1 - e^{-α t})) * exp(-ka t) * exp(-km m) * exp(c t)
    """

    V0: float     # initial reference value (MSRP-like)
    delta: float  # early drop magnitude (0-1)
    alpha: float  # early drop speed (>0)
    k_a: float    # long-term age decay rate (>=0)
    k_m: float    # mileage decay rate (>=0)
    L: float      # value floor (>=0)
    c: float      # appreciation rate (>=0)


def unconstrained_to_params(theta: np.ndarray) -> UniversalParams:
    """
    Map unconstrained parameters (in R^7) to constrained UniversalParams.

    We use:
    - softplus(x) = log(1 + exp(x)) for strictly positive params
    - sigmoid(x)  = 1 / (1 + exp(-x)) for [0,1] params
    """

    assert theta.shape[-1] == 7, "theta must have 7 components"

    def softplus(x: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(x))

    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    # unpack unconstrained
    t_V0, t_delta, t_alpha, t_ka, t_km, t_L, t_c = theta

    V0 = softplus(t_V0) + 1e-3   # > 0
    delta = sigmoid(t_delta)     # in (0,1)
    alpha = softplus(t_alpha)    # > 0
    k_a = softplus(t_ka)         # >= 0
    k_m = softplus(t_km)         # >= 0
    L = softplus(t_L)            # >= 0
    c = softplus(t_c)            # >= 0

    return UniversalParams(
        V0=float(V0),
        delta=float(delta),
        alpha=float(alpha),
        k_a=float(k_a),
        k_m=float(k_m),
        L=float(L),
        c=float(c),
    )


def value_universal(
    t: np.ndarray,
    m: np.ndarray,
    params: UniversalParams,
) -> np.ndarray:
    """
    Compute V(t, m) for vectors/arrays t (age in years) and m (mileage unit, e.g. in 10k km).

    t and m should be broadcastable to the same shape.
    """
    t = np.asarray(t, dtype=float)
    m = np.asarray(m, dtype=float)

    V0 = params.V0
    delta = params.delta
    alpha = params.alpha
    k_a = params.k_a
    k_m = params.k_m
    L = params.L
    c = params.c

    # early drop factor
    early_factor = 1.0 - delta * (1.0 - np.exp(-alpha * t))

    # age decay
    age_factor = np.exp(-k_a * t)

    # mileage decay
    mileage_factor = np.exp(-k_m * m)

    # appreciation factor
    appreciation_factor = np.exp(c * t)

    return L + (V0 - L) * early_factor * age_factor * mileage_factor * appreciation_factor


def value_universal_from_unconstrained(
    t: np.ndarray,
    m: np.ndarray,
    theta: np.ndarray,
) -> Tuple[np.ndarray, UniversalParams]:
    """
    Convenience wrapper:
    - converts unconstrained theta to UniversalParams
    - returns both the values and the params object
    """
    params = unconstrained_to_params(theta)
    v = value_universal(t, m, params)
    return v, params
