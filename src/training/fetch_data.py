# src/training/fetch_data.py

from typing import List, Dict, Any
from src.data.supabase_client import get_supabase_client


def fetch_training_rows_since(since_date: str) -> List[Dict[str, Any]]:
    """
    Fetch joined timeseries + vehicles + gammes rows with date >= since_date.
    since_date: 'YYYY-MM-DD'
    """
    client = get_supabase_client()

    resp = (
        client.table("vehicle_timeseries")
        .select(
            # Base timeseries fields
            "vehicle_id, date, value, mileage,"
            # Join through vehicles -> and inside that, join gammes
            "vehicles!inner("
            "  vehicle_id, gamme_id, type, color, transmission, fuel_type, drivetrain,"
            "  bodystyle, interior_color, paddock_score," #aftermarket_mods,features
            "  city, state, doors_numbers,"
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

    return resp.data or []
