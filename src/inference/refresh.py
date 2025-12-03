from __future__ import annotations

from datetime import date
from typing import List, Dict, Any

from src.data.supabase_client import (
    get_supabase_client,
    upsert_vehicle_timeseries_point,
)
from src.inference.valuation_service import ValuatorEngine


def refresh_user_vehicles(user_id: str) -> List[Dict[str, Any]]:
    """
    For a given user_id:
      - Fetch all their vehicles
      - For each vehicle, check the latest entry in vehicle_timeseries
      - If already valued today -> skip
      - Otherwise -> run ValuatorEngine and upsert a new timeseries point
      - Also update vehicles.current_value with the new valuation

    Returns a list of dicts with:
      {
        "vehicle_id": ...,
        "price_usd": ...,
        "comment": ...,
      }
    """

    client = get_supabase_client()
    engine = ValuatorEngine()
    today = date.today()
    today_str = today.isoformat()

    # 1) Fetch all vehicles for this user
    vehicles_resp = (
        client.table("vehicles")
        .select("*")
        .eq("user_id", user_id)
        .execute()
    )
    vehicles = vehicles_resp.data or []

    results: List[Dict[str, Any]] = []

    for v in vehicles:
        vehicle_id = v["vehicle_id"]

        # 2) Check latest timeseries entry for this vehicle
        ts_resp = (
            client.table("vehicle_timeseries")
            .select("date")
            .eq("vehicle_id", vehicle_id)
            .order("date", desc=True)
            .limit(1)
            .execute()
        )
        ts_rows = ts_resp.data or []
        latest_date_str = ts_rows[0]["date"] if ts_rows else None

        if latest_date_str == today_str:
            # Already valued today -> skip to next vehicle
            continue

        # 3) Run valuation for this vehicle
        valuation = engine.evaluate(v)

        # 4) Upsert into vehicle_timeseries
        mileage = v.get("mileage")

        upsert_vehicle_timeseries_point(
            vehicle_id=vehicle_id,
            value=valuation.price_usd,
            mileage=mileage,
            date=today,
        )

        # 5) Update vehicles.current_value with the new price
        (
            client.table("vehicles")
            .update({"current_value": valuation.price_usd})
            .eq("vehicle_id", vehicle_id)
            .execute()
        )

        results.append(
            {
                "vehicle_id": vehicle_id,
                "price_usd": valuation.price_usd,
                "comment": valuation.comment,
            }
        )

    return results
