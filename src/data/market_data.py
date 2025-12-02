# src/data/market_data.py

from __future__ import annotations

from typing import List, Dict, Any, Optional
from datetime import datetime

import numpy as np

from src.data.supabase_client import get_supabase_client


def _parse_date(d: str) -> datetime:
    """
    Parse ISO date/datetime string from Supabase.
    Handles 'YYYY-MM-DD' and 'YYYY-MM-DDTHH:MM:SSZ' styles.
    """
    try:
        return datetime.fromisoformat(d.replace("Z", "+00:00"))
    except Exception:
        # fallback: date only
        return datetime.strptime(d[:10], "%Y-%m-%d")


def _compute_age_years(model_year: int, dt: datetime) -> float:
    """
    Approximate age t in years from model_year and valuation datetime.

    We take t = (dt - Jan 1 of model_year) / 365.25.
    """
    start = datetime(model_year, 1, 1, tzinfo=dt.tzinfo)
    delta_days = (dt - start).days
    return delta_days / 365.25


def fetch_points_for_car_market(
    make: str,
    model: str,
    year: int,
    trim: Optional[str],
    max_vehicles: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Build training data for ONE Car Market:
      CM = {make, model, year, trim}

    We:
      1) Find all vehicles in `vehicles` matching that CM.
      2) Fetch all their rows from `vehicle_timeseries`.
      3) Convert them into arrays of:
           - t: age in years
           - m: mileage (as stored)
           - v: observed value

    Returns:
      {
        "t": np.ndarray,
        "m": np.ndarray,
        "v": np.ndarray,
        "n_points": int
      }
    """
    client = get_supabase_client()

    # 1) Find matching vehicles
    q = (
        client.table("vehicles")
        .select("id, make, model, year, trim")
        .eq("make", make)
        .eq("model", model)
        .eq("year", year)
    )

    if trim is not None:
        q = q.eq("trim", trim)
    else:
        q = q.is_("trim", None)

    vehicles_resp = q.execute()
    vehicles = vehicles_resp.data or []

    if not vehicles:
        return {"t": np.array([]), "m": np.array([]), "v": np.array([]), "n_points": 0}

    if max_vehicles is not None and len(vehicles) > max_vehicles:
        vehicles = vehicles[:max_vehicles]

    vehicle_by_id = {v["id"]: v for v in vehicles}
    vehicle_ids = list(vehicle_by_id.keys())

    # 2) Fetch timeseries for these vehicles
    ts_query = (
        client.table("vehicle_timeseries")
        .select("vehicle_id, date, mileage, value")
        .in_("vehicle_id", vehicle_ids)
    )
    ts_resp = ts_query.execute()
    ts_rows = ts_resp.data or []

    if not ts_rows:
        return {"t": np.array([]), "m": np.array([]), "v": np.array([]), "n_points": 0}

    t_list: List[float] = []
    m_list: List[float] = []
    v_list: List[float] = []

    for row in ts_rows:
        vid = row["vehicle_id"]
        vehicle = vehicle_by_id.get(vid)
        if vehicle is None:
            continue

        model_year = vehicle["year"]
        date_str = row["date"]
        dt = _parse_date(date_str)

        age_years = _compute_age_years(model_year, dt)
        mileage = row.get("mileage")
        value = row.get("value")

        if value is None:
            continue

        if mileage is None:
            mileage = 0.0

        t_list.append(float(age_years))
        m_list.append(float(mileage))
        v_list.append(float(value))

    if not t_list:
        return {"t": np.array([]), "m": np.array([]), "v": np.array([]), "n_points": 0}

    t_arr = np.array(t_list, dtype=float)
    m_arr = np.array(m_list, dtype=float)
    v_arr = np.array(v_list, dtype=float)

    return {
        "t": t_arr,
        "m": m_arr,
        "v": v_arr,
        "n_points": len(t_arr),
    }
