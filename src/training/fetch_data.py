# src/training/fetch_data.py

from __future__ import annotations

from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import requests

from src.data.supabase_client import get_supabase_client

# Base URLs for MarketCheck "comparables" + "MarketCheck Price™"
MC_BASE_PREDICT = "https://api.marketcheck.com/v2/predict/car/us/marketcheck_price"
MC_BASE_COMPARABLES = f"{MC_BASE_PREDICT}/comparables"


def _get_marketcheck_api_key() -> Optional[str]:
    return os.getenv("MARKETCHECK_API_KEY") or os.getenv("VITE_MARKETCHECK_API_KEY")


def _fetch_mc_comparables_for_vehicle(
    base_row: Dict[str, Any],
    max_similars: int = 5,
) -> List[Dict[str, Any]]:

    api_key = _get_marketcheck_api_key()
    if not api_key:
        print("[MC] No MARKETCHECK_API_KEY configured; skipping ALL comparables.")
        return []

    vehicles = base_row.get("vehicles") or {}
    gammes = (vehicles or {}).get("gammes") or {}

    vehicle_id = base_row.get("vehicle_id")
    vin = vehicles.get("vin")
    mileage = base_row.get("mileage")
    dealer_type = vehicles.get("dealer_type") or "used"

    city = vehicles.get("city")
    state = vehicles.get("state")
    zip_code = vehicles.get("zip")

    make = gammes.get("make")
    model = gammes.get("model")
    year = gammes.get("year")
    trim = gammes.get("trim")

    # --- Required checks ---
    if not vin:
        print(f"[MC] Skipping vehicle (no VIN). vehicle_id={vehicle_id}, vin={vin}")
        return []

    if mileage is None:
        print(f"[MC] Skipping vehicle (no mileage). vehicle_id={vehicle_id}, vin={vin}")
        return []

    if not (zip_code or (city and state)):
        print(
            f"[MC] Skipping vehicle (no valid location). vehicle_id={vehicle_id}, "
            f"vin={vin}, city={city}, state={state}, zip={zip_code}"
        )
        return []

    if not (make and model and year):
        print(
            f"[MC] Skipping vehicle (missing make/model/year). vehicle_id={vehicle_id}, "
            f"vin={vin}, make={make}, model={model}, year={year}"
        )
        return []

    # Build params
    params = {
        "api_key": api_key,
        "vin": vin,
        "miles": mileage,
        "dealer_type": dealer_type,
        "rows": max_similars,
        "make": make,
        "model": model,
        "year": year,
    }

    if trim:
        params["trim"] = trim

    if zip_code:
        params["zip"] = zip_code
    else:
        params["city"] = city
        params["state"] = state

    # 🔍 NEW DEBUG LOG — prints everything sent to MC
    print(
        f"[MC] Calling comparables for vehicle_id={vehicle_id}, vin={vin}. "
        f"Params={params}"
    )

    # Actual API request
    try:
        resp = requests.get(MC_BASE_COMPARABLES, params=params, timeout=10)
        if resp.status_code != 200:
            print(
                f"[MC] comparables error {resp.status_code} for vehicle_id={vehicle_id}, "
                f"vin={vin}: {resp.text[:200]}"
            )
            return []

        data = resp.json()
        listings = data.get("listings", data if isinstance(data, list) else [])

        simplified = []
        for lst in listings[:max_similars]:
            build = lst.get("build", {})
            sim_make = build.get("make") or lst.get("make") or make
            sim_model = build.get("model") or lst.get("model") or model
            sim_year = build.get("year") or lst.get("year") or year
            sim_trim = build.get("trim") or lst.get("trim") or trim
            sim_miles = lst.get("miles") or lst.get("mileage") or mileage
            sim_id = lst.get("id") or lst.get("listing_id")

            simplified.append(
                {
                    "listing_id": sim_id,
                    "make": sim_make,
                    "model": sim_model,
                    "year": int(sim_year),
                    "trim": sim_trim,
                    "mileage": float(sim_miles) if sim_miles is not None else 0.0,
                    "raw": lst,
                }
            )

        return simplified

    except Exception as e:
        print(
            f"[MC] comparables exception for vehicle_id={vehicle_id}, vin={vin}: {e}"
        )
        return []


def _fetch_mc_price_for_spec(spec: Dict[str, Any]) -> Optional[float]:

    api_key = _get_marketcheck_api_key()
    if not api_key:
        return None

    listing_id = spec.get("listing_id")

    params = {
        "api_key": api_key,
        "make": spec["make"],
        "model": spec["model"],
        "year": spec["year"],
    }

    if spec.get("trim"):
        params["trim"] = spec["trim"]
    if spec.get("mileage") is not None:
        params["miles"] = spec["mileage"]
    if spec.get("state"):
        params["state"] = spec["state"]
    if spec.get("city"):
        params["city"] = spec["city"]

    try:
        resp = requests.get(MC_BASE_PREDICT, params=params, timeout=10)
        if resp.status_code != 200:
            print(
                f"[MC] price error {resp.status_code} for listing_id={listing_id}, "
                f"make={spec.get('make')}, model={spec.get('model')}, "
                f"year={spec.get('year')}: {resp.text[:200]}"
            )
            return None

        data = resp.json()
        price = (
            data.get("predicted_price")
            or data.get("marketcheck_price")
            or data.get("price")
        )
        if price is None:
            print(
                f"[MC] price response missing price field for listing_id={listing_id}, spec={spec}"
            )
            return None

        return float(price)

    except Exception as e:
        print(
            f"[MC] price exception for listing_id={listing_id}, "
            f"make={spec.get('make')}, model={spec.get('model')}, "
            f"year={spec.get('year')}: {e}"
        )
        return None


def _build_mc_training_row(base_row, spec, mc_price):

    vehicles = base_row.get("vehicles") or {}
    gammes = (vehicles or {}).get("gammes") or {}

    today = datetime.utcnow().date().isoformat()

    return {
        "vehicle_id": f"mc_{spec.get('listing_id') or spec.get('make')}_{spec.get('year')}",
        "date": today,
        "value": mc_price,
        "mileage": spec.get("mileage", 0.0),
        "source": "marketcheck",

        "vehicles": {
            "vehicle_id": None,
            "gamme_id": vehicles.get("gamme_id"),
            "type": vehicles.get("type"),
            "color": vehicles.get("color"),
            "transmission": vehicles.get("transmission"),
            "fuel_type": vehicles.get("fuel_type"),
            "drivetrain": vehicles.get("drivetrain"),
            "bodystyle": vehicles.get("bodystyle"),
            "interior_color": vehicles.get("interior_color"),
            "paddock_score": vehicles.get("paddock_score"),
            "city": vehicles.get("city"),
            "state": vehicles.get("state"),
            "doors_numbers": vehicles.get("doors_numbers"),

            "gammes": {
                "gamme_id": gammes.get("gamme_id"),
                "make": spec["make"],
                "model": spec["model"],
                "year": spec["year"],
                "trim": spec.get("trim"),
                "V0": gammes.get("V0"),
                "hp": gammes.get("hp"),
                "engine_displacement": gammes.get("engine_displacement"),
                "engine_configuration": gammes.get("engine_configuration"),
                "number_of_cylinders": gammes.get("number_of_cylinders"),
                "supply_by_country": gammes.get("supply_by_country"),
                "delta": gammes.get("delta"),
                "alpha": gammes.get("alpha"),
                "k_a": gammes.get("k_a"),
                "k_m": gammes.get("k_m"),
                "L": gammes.get("L"),
                "c": gammes.get("c"),
            },
        },
    }


def fetch_training_rows_since(
    since_date: str,
    use_marketcheck: bool = True,
    max_mc_vehicles: int = 50,
    max_similars_per_vehicle: int = 5,
) -> List[Dict[str, Any]]:

    client = get_supabase_client()

    resp = (
        client.table("vehicle_timeseries")
        .select(
            "vehicle_id, date, value, mileage,"
            "vehicles!inner("
            "  vehicle_id, gamme_id, vin, type, color, transmission, fuel_type, drivetrain,"
            "  bodystyle, interior_color, paddock_score,"
            "  city, state, zip, dealer_type, doors_numbers,"
            "  gammes!inner("
            "    gamme_id, make, model, year, trim, V0, hp, engine_displacement,"
            "    engine_configuration, number_of_cylinders, supply_by_country,"
            "    delta, alpha, k_a, k_m, L, c"
            "  )"
            ")"
        )
        .gte("date", since_date)
        .execute()
    )

    base_rows = resp.data or []
    for r in base_rows:
        r["source"] = "supabase"

    if not use_marketcheck:
        return base_rows

    mc_rows = []

    for idx, base_row in enumerate(base_rows):
        if idx >= max_mc_vehicles:
            break

        vehicles = base_row.get("vehicles") or {}
        vin = vehicles.get("vin")
        print(f"[MC] Processing idx={idx}, vehicle_id={base_row.get('vehicle_id')}, vin={vin}")

        comparables = _fetch_mc_comparables_for_vehicle(
            base_row,
            max_similars=max_similars_per_vehicle,
        )

        if not comparables:
            print(f"[MC] No comparables for vehicle_id={base_row.get('vehicle_id')}, vin={vin}")
            continue

        city = vehicles.get("city")
        state = vehicles.get("state")

        for comp in comparables:
            spec = {
                "listing_id": comp.get("listing_id"),
                "make": comp["make"],
                "model": comp["model"],
                "year": comp["year"],
                "trim": comp.get("trim"),
                "mileage": comp.get("mileage", 0.0),
                "city": city,
                "state": state,
            }

            mc_price = _fetch_mc_price_for_spec(spec)
            if mc_price is None:
                continue

            mc_row = _build_mc_training_row(base_row, spec, mc_price)
            mc_rows.append(mc_row)

    return base_rows + mc_rows
