# src/parametric/fit_market_params.py

from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np

from scipy.optimize import minimize

from src.data.supabase_client import get_supabase_client
from src.data.market_data import fetch_points_for_car_market
from src.data.marketcheck_client import fetch_marketcheck_points_for_car_market

from src.parametric.depreciation_function import (
    UniversalParams,
    value_universal,
)


TABLE_NAME = "vehicles_market_nn_parameters"


def _build_params_from_unconstrained(theta6: np.ndarray, V0: float) -> UniversalParams:
    """
    Map unconstrained 6-d vector to constrained params, keeping V0 fixed.

    theta6 = [t_delta, t_alpha, t_k_a, t_k_m, t_L, t_c]
    """
    assert theta6.shape[-1] == 6

    def softplus(x: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(x))

    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    t_delta, t_alpha, t_k_a, t_k_m, t_L, t_c = theta6

    delta = sigmoid(t_delta)
    alpha = softplus(t_alpha)
    k_a = softplus(t_k_a)
    k_m = softplus(t_k_m)
    L = softplus(t_L)
    c = softplus(t_c)

    return UniversalParams(
        V0=V0,
        delta=float(delta),
        alpha=float(alpha),
        k_a=float(k_a),
        k_m=float(k_m),
        L=float(L),
        c=float(c),
    )


def fit_params_for_car_market(
    make: str,
    model: str,
    year: int,
    trim: Optional[str],
    use_marketcheck: bool = True,
    max_vehicles: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Fit the 6 PDM parameters for a Car Market CM = {make, model, year, trim}.

    Steps:
      1) Read V0 (MSRP) from vehicles_market_nn_parameters.
      2) Aggregate (t, m, v) from Supabase (vehicle_timeseries).
      3) Optionally add MarketCheck (t, m, v) at t ~ today.
      4) Optimize a 6D unconstrained vector to minimize MSE between
         V_CM(t, m; params) and v.
      5) Update vehicles_market_nn_parameters with the fitted params.

    Returns:
      {
        "status": "ok" | "no_data" | "error",
        "n_points": int,
        "params": UniversalParams or None,
        "loss": float or None,
        "row": dict or None,
      }
    """
    client = get_supabase_client()

    # 1) Get Market Car row and V0
    q = (
        client.table(TABLE_NAME)
        .select("*")
        .eq("make", make)
        .eq("model", model)
        .eq("year", year)
    )
    if trim is not None:
        q = q.eq("trim", trim)
    else:
        q = q.is_("trim", None)

    resp = q.limit(1).execute()
    rows = resp.data or []

    if not rows:
        return {"status": "error", "message": "No market row found. Run MSRP extraction first.", "n_points": 0, "params": None, "loss": None, "row": None}

    row = rows[0]
    row_id = row["vehicle_market_id"]
    V0 = row.get("V0")

    if V0 is None:
        return {"status": "error", "message": "V0 (MSRP) missing. Fill it first.", "n_points": 0, "params": None, "loss": None, "row": row}

    # 2) Get Supabase points
    supa_data = fetch_points_for_car_market(make, model, year, trim, max_vehicles=max_vehicles)
    t_s, m_s, v_s = supa_data["t"], supa_data["m"], supa_data["v"]

    t_list = [t_s]
    m_list = [m_s]
    v_list = [v_s]

    # 3) Optionally add MarketCheck points
    if use_marketcheck:
        mc_data = fetch_marketcheck_points_for_car_market(make, model, year, trim, max_results=50)
        if mc_data["n_points"] > 0:
            t_list.append(mc_data["t"])
            m_list.append(mc_data["m"])
            v_list.append(mc_data["v"])

    # Combine
    t = np.concatenate([arr for arr in t_list if arr.size > 0], axis=0)
    m = np.concatenate([arr for arr in m_list if arr.size > 0], axis=0)
    v = np.concatenate([arr for arr in v_list if arr.size > 0], axis=0)

    if t.size == 0:
        return {"status": "no_data", "message": "No training points found for this Car Market.", "n_points": 0, "params": None, "loss": None, "row": row}

    # 4) Optimize the 6 unconstrained params
    def loss_fn(theta6: np.ndarray) -> float:
        params = _build_params_from_unconstrained(theta6, V0)
        v_pred = value_universal(t, m, params)
        return float(np.mean((v_pred - v) ** 2))

    theta0 = np.zeros(6, dtype=float)
    res = minimize(loss_fn, theta0, method="BFGS")

    theta_opt = res.x
    params_opt = _build_params_from_unconstrained(theta_opt, V0)
    final_loss = loss_fn(theta_opt)

    # 5) Update DB
    update_payload = {
        "delta": params_opt.delta,
        "alpha": params_opt.alpha,
        "k_a": params_opt.k_a,
        "k_m": params_opt.k_m,
        "L": params_opt.L,
        "c": params_opt.c,
        "fit_loss": final_loss,
        "data_points": int(t.size),
    }

    update_resp = (
        client.table(TABLE_NAME)
        .update(update_payload)
        .eq("vehicle_market_id", row_id)
        .execute()
    )
    updated_row = update_resp.data[0] if update_resp.data else row

    return {
        "status": "ok",
        "n_points": int(t.size),
        "params": params_opt,
        "loss": final_loss,
        "row": updated_row,
    }
