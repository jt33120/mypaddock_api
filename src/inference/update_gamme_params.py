# src/inference/update_gamme_params.py

from __future__ import annotations

from typing import Dict, Any

import torch

from src.data.supabase_client import get_supabase_client
from src.training.encoding import encode_sample
from src.training.incremental_trainer import ParamsNN, MODEL_PATH


def _load_trained_model() -> ParamsNN:
    """
    Load the trained ParamsNN from disk in eval mode.
    """
    model = ParamsNN()
    if MODEL_PATH.exists():
        state = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state)
    model.eval()
    return model


def update_gamme_params(gamme_id: str) -> Dict[str, Any]:
    """
    For a given gamme_id:
      - Fetch one representative vehicle + its gamme row from Supabase
      - Build the full feature vector using encode_sample(...)
      - Run the ParamsNN to get [delta, alpha, k_a, k_m, L, c]
      - Update the corresponding row in 'gammes' table.

    Returns a small dict with status and the new params.
    """
    client = get_supabase_client()

    # 1) Fetch one vehicle belonging to this gamme, with nested gamme info
    resp = (
        client.table("vehicles")
        .select(
            "vehicle_id, gamme_id, type, color, transmission, fuel_type, drivetrain,"
            "bodystyle, interior_color, features, paddock_score, aftermarket_mods,"
            "city, state, doors_numbers,"
            "gammes!inner("
            "  gamme_id, make, model, year, trim, V0, hp, engine_displacement,"
            "  engine_configuration, number_of_cylinders, supply_by_country,"
            "  delta, alpha, k_a, k_m, L, c"
            ")"
        )
        .eq("gamme_id", gamme_id)
        .limit(1)
        .execute()
    )

    data = resp.data or []
    if not data:
        return {
            "status": "no_vehicle_for_gamme",
            "gamme_id": gamme_id,
        }

    row = data[0]
    v = row
    g = row["gammes"]

    # 2) Build feature vector
    nn_input = encode_sample(v, g).unsqueeze(0)  # [1, INPUT_DIM]

    # 3) Load model and predict params
    model = _load_trained_model()
    with torch.no_grad():
        params_tensor = model(nn_input)[0]  # [6]

    delta, alpha, k_a, k_m, L, c = [float(x) for x in params_tensor.tolist()]

    # 4) Update the gammes table with the new parameters
    update_resp = (
        client.table("gammes")
        .update(
            {
                "delta": delta,
                "alpha": alpha,
                "k_a": k_a,
                "k_m": k_m,
                "L": L,
                "c": c,
            }
        )
        .eq("gamme_id", gamme_id)
        .execute()
    )

    return {
        "status": "ok",
        "gamme_id": gamme_id,
        "params": {
            "delta": delta,
            "alpha": alpha,
            "k_a": k_a,
            "k_m": k_m,
            "L": L,
            "c": c,
        },
        "supabase_update": update_resp.data,
    }
