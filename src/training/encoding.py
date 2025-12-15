# src/training/encoding.py
import torch
from typing import Dict, Any
import hashlib


def _stable_hash_to_unit_float(x: str) -> float:
    """
    Stable hash -> [0, 1). Unlike Python's built-in hash(), this is stable across runs.
    """
    if not x:
        return 0.0
    h = hashlib.md5(x.encode("utf-8")).hexdigest()  # stable
    # take first 8 hex chars -> 32-bit int
    val = int(h[:8], 16)
    return (val % 10_000) / 10_000.0


def encode_sample(v: Dict[str, Any], g: Dict[str, Any]) -> torch.Tensor:
    # ---- Gamme numeric fields ----
    year = float(g.get("year") or 0.0)
    V0 = float(g.get("V0") or 0.0)
    hp = float(g.get("hp") or 0.0)
    disp = float(g.get("engine_displacement") or 0.0)
    cylinders = float(g.get("number_of_cylinders") or 0.0)

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

    make_h = _stable_hash_to_unit_float(make)
    model_h = _stable_hash_to_unit_float(model)
    trim_h = _stable_hash_to_unit_float(trim)
    engine_conf_h = _stable_hash_to_unit_float(engine_conf)

    # ---- Vehicle numeric-ish fields ----
    paddock_score = float(v.get("paddock_score") or 50.0)
    doors_num = float(v.get("doors_numbers") or 0.0)

    features_raw = v.get("features") or []
    features_count = float(len(features_raw)) if isinstance(features_raw, list) else 0.0

    mods_raw = v.get("aftermarket_mods") or []
    mods_count = float(len(mods_raw)) if isinstance(mods_raw, list) else 0.0

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

    v_type_h = _stable_hash_to_unit_float(v_type)
    color_h = _stable_hash_to_unit_float(color)
    transmission_h = _stable_hash_to_unit_float(transmission)
    fuel_type_h = _stable_hash_to_unit_float(fuel_type)
    drivetrain_h = _stable_hash_to_unit_float(drivetrain)
    bodystyle_h = _stable_hash_to_unit_float(bodystyle)
    interior_color_h = _stable_hash_to_unit_float(interior_color)
    city_h = _stable_hash_to_unit_float(city)
    state_h = _stable_hash_to_unit_float(state)

    features = [
        # Gamme numeric
        year,
        V0,
        hp,
        disp,
        cylinders,
        total_supply,

        # Gamme categorical (stable hashed)
        make_h,
        model_h,
        trim_h,
        engine_conf_h,

        # Vehicle numeric
        paddock_score,
        doors_num,
        features_count,
        mods_count,

        # Vehicle categorical (stable hashed)
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
