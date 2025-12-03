# src/training/encoding.py

import torch
from typing import Dict, Any


def _hash_to_unit_float(x: str) -> float:
    """
    Simple hash -> [0, 1) float.
    Not fancy, but lets us inject categorical information quickly.
    """
    if not x:
        return 0.0
    # Use Python's hash, make it positive, mod a large number, normalize
    h = hash(x)
    h = h if h >= 0 else -h
    return (h % 10_000) / 10_000.0


def encode_sample(v: Dict[str, Any], g: Dict[str, Any]) -> torch.Tensor:
    """
    Encode *all* relevant gamme + vehicle fields into a numeric vector
    suitable as input for ParamsNN.

    v: row["vehicles"]
    g: row["gammes"]
    """

    # ---- Gamme numeric fields ----
    year = float(g.get("year") or 0.0)
    V0 = float(g.get("V0") or 0.0)
    hp = float(g.get("hp") or 0.0)
    disp = float(g.get("engine_displacement") or 0.0)
    cylinders = float(g.get("number_of_cylinders") or 0.0)

    # supply_by_country: sum over all countries as a rough proxy
    supply_raw = g.get("supply_by_country") or {}
    total_supply = 0.0
    if isinstance(supply_raw, dict):
        try:
            total_supply = float(sum(supply_raw.values()) or 0.0)
        except Exception:
            total_supply = 0.0

    # ---- Gamme categorical fields ----
    make = str(g.get("make") or "")
    model = str(g.get("model") or "")
    trim = str(g.get("trim") or "")
    engine_conf = str(g.get("engine_configuration") or "")

    make_h = _hash_to_unit_float(make)
    model_h = _hash_to_unit_float(model)
    trim_h = _hash_to_unit_float(trim)
    engine_conf_h = _hash_to_unit_float(engine_conf)

    # ---- Vehicle numeric-ish fields ----
    paddock_score = float(v.get("paddock_score") or 50.0)
    doors_num = float(v.get("doors_numbers") or 0.0)

    # features: text[] -> length
    features_raw = v.get("features") or []
    if isinstance(features_raw, list):
        features_count = float(len(features_raw))
    else:
        features_count = 0.0

    # aftermarket_mods: text[] -> length
    mods_raw = v.get("aftermarket_mods") or []
    if isinstance(mods_raw, list):
        mods_count = float(len(mods_raw))
    else:
        mods_count = 0.0

    # ---- Vehicle categorical fields ----
    v_type = str(v.get("type") or "")
    color = str(v.get("color") or "")
    transmission = str(v.get("transmission") or "")
    fuel_type = str(v.get("fuel_type") or "")
    drivetrain = str(v.get("drivetrain") or "")
    bodystyle = str(v.get("bodystyle") or "")
    interior_color = str(v.get("interior_color") or "")
    city = str(v.get("city") or "")
    state = str(v.get("state") or "")

    v_type_h = _hash_to_unit_float(v_type)
    color_h = _hash_to_unit_float(color)
    transmission_h = _hash_to_unit_float(transmission)
    fuel_type_h = _hash_to_unit_float(fuel_type)
    drivetrain_h = _hash_to_unit_float(drivetrain)
    bodystyle_h = _hash_to_unit_float(bodystyle)
    interior_color_h = _hash_to_unit_float(interior_color)
    city_h = _hash_to_unit_float(city)
    state_h = _hash_to_unit_float(state)

    # Concatenate everything into a single feature vector
    # You can reorder/extend this, but ALL the info is now present.
    features = [
        # Gamme numeric
        year,
        V0,
        hp,
        disp,
        cylinders,
        total_supply,

        # Gamme categorical (hashed)
        make_h,
        model_h,
        trim_h,
        engine_conf_h,

        # Vehicle numeric
        paddock_score,
        doors_num,
        features_count,
        mods_count,

        # Vehicle categorical (hashed)
        v_type_h,
        color_h,
        transmission_h,
        fuel_type_h,
        drivetrain_h,
        bodystyle_h,
        interior_color_h,
        city_h,
        state_h,
    ]

    return torch.tensor(features, dtype=torch.float32)
