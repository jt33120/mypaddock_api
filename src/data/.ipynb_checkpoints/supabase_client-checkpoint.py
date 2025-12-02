from typing import List, Dict, Optional, Any
from datetime import date
from supabase import create_client, Client

from src.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY

_supabase_client: Client | None = None


def get_supabase_client() -> Client:
    """
    Singleton-style accessor for the Supabase client.
    """
    global _supabase_client

    if _supabase_client is None:
        if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
            raise RuntimeError(
                "SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY is missing. "
                "Check your .env file."
            )
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

    return _supabase_client


def fetch_vehicles(limit: int = 500) -> List[Dict]:
    """
    Fetch up to `limit` vehicle rows from the 'vehicles' table.
    Change the table name if yours is different.
    """
    client = get_supabase_client()
    response = client.table("vehicles").select("*").limit(limit).execute()
    return response.data

def fetch_vehicle_by_id(vehicle_id: str, table_name: str = "vehicles") -> Optional[Dict]:
    client = get_supabase_client()
    response = client.table(table_name).select("*").eq("vehicle_id", vehicle_id).limit(1).execute()
    rows = response.data or []
    return rows[0] if rows else None

def upsert_vehicle_timeseries_point(
    vehicle_id: str,
    value: float,
    mileage: Optional[float] = None,
    at_date: Optional[date] = None,
    table_name: str = "vehicle_timeseries",
) -> Dict[str, Any]:
    """
    Insert or update a timeseries point for a given vehicle and date.

    Behavior:
    - If a row already exists for (vehicle_id, date) -> update it
    - Otherwise -> insert a new row

    Assumes:
    - `vehicle_timeseries` has at least: id, vehicle_id, value, mileage, date
    - `date` is a DATE column (no time) or compatible

    Returns a small dict describing what happened.
    """
    from datetime import date as _date  # avoid shadowing

    client = get_supabase_client()
    if at_date is None:
        at_date = _date.today()

    # Make sure we use ISO format 'YYYY-MM-DD' for the query
    date_str = at_date.isoformat()

    # 1. Check if a row already exists for this vehicle and date
    check_resp = client.table(table_name) \
        .select("vehicle_id") \
        .eq("vehicle_id", vehicle_id) \
        .eq("date", date_str) \
        .limit(1) \
        .execute()

    existing_rows = check_resp.data or []

    if existing_rows:
        # 2. Update existing row
        row_id = existing_rows[0]["vehicle_id"]

        update_payload = {
            "value": value,
            "mileage": mileage,
            "date": date_str,
        }

        update_resp = client.table(table_name) \
            .update(update_payload) \
            .eq("vehicle_id", row_id) \
            .execute()

        return {
            "action": "updated",
            "row_id": row_id,
            "response": update_resp.data,
        }

    else:
        # 3. Insert new row
        insert_payload = {
            "vehicle_id": vehicle_id,
            "value": value,
            "mileage": mileage,
            "date": date_str,
        }

        insert_resp = client.table(table_name) \
            .insert(insert_payload) \
            .execute()

        # Assuming Supabase returns the inserted row
        new_id = insert_resp.data[0]["vehicle_id"] if insert_resp.data else None

        return {
            "action": "inserted",
            "row_id": new_id,
            "response": insert_resp.data,
        }


