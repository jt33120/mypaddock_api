# src/data/marketcheck_client.py

from __future__ import annotations

from typing import Dict, Any, Optional

from datetime import datetime

import os
import requests
import numpy as np

from src.config import MARKETCHECK_API_KEY  # make sure you define this in config.py


BASE_URL = "https://marketcheck-prod.apigee.net/v2/search/car/active"


def fetch_marketcheck_points_for_car_market(
    make: str,
    model: str,
    year: int,
    trim: Optional[str],
    max_results: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Query MarketCheck for active listings matching this Car Market:

      CM = {make, model, year, trim}

    and return (t, m, v) points where:
      - t: age in years (all roughly 'today' - Jan 1st model_year)
      - m: mileage from listing
      - v: listing_price

    If API key is missing or something fails, returns empty arrays.
    """
    api_key = MARKETCHECK_API_KEY or os.getenv("MARKETCHECK_API_KEY")
    if not api_key:
        return {"t": np.array([]), "m": np.array([]), "v": np.array([]), "n_points": 0}

    params = {
        "api_key": api_key,
        "make": make,
        "model": model,
        "year": year,
        "car_type": "used",
        "rows": max_results,
        "start": 0,
    }
    if trim:
        params["trim"] = trim

    try:
        resp = requests.get(BASE_URL, params=params, timeout=10)
        if resp.status_code != 200:
            return {"t": np.array([]), "m": np.array([]), "v": np.array([]), "n_points": 0}

        data = resp.json()
        listings = data.get("listings", [])

        t_list = []
        m_list = []
        v_list = []

        now = datetime.utcnow()
        for lst in listings:
            price = lst.get("price")
            mileage = lst.get("miles") or lst.get("mileage")

            if price is None:
                continue
            if mileage is None:
                mileage = 0.0

            # Age t is effectively "today - model_year"
            model_year = year  # by definition of the query
            age_years = (now - datetime(model_year, 1, 1)).days / 365.25

            t_list.append(float(age_years))
            m_list.append(float(mileage))
            v_list.append(float(price))

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

    except Exception:
        return {"t": np.array([]), "m": np.array([]), "v": np.array([]), "n_points": 0}
