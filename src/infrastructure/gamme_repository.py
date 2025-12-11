# infrastructure/gamme_repository.py

from typing import List, Dict, Any

from supabase import create_client, Client

from core.domain import VehicleInfo, MaintenanceTask
from config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY


def _build_gamme_id(vehicle: VehicleInfo) -> str:
    """
    Build a stable gamme_id like: BMW3SERIES328IXDRIVESPORTPACKAGE2013
    """
    parts = [
        (vehicle.make or "").upper().replace(" ", ""),
        (vehicle.model or "").upper().replace(" ", ""),
        (vehicle.trim or "").upper().replace(" ", ""),
        str(vehicle.year),
    ]
    return "".join(parts)


def _get_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Supabase URL or SERVICE_ROLE key not set in config.")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def upsert_gamme_schedule(
    vehicle: VehicleInfo,
    vehicle_type: str,
    schedule: List[MaintenanceTask],
) -> Dict[str, Any]:
    """
    Upsert the maintenance schedule for a given gamme into Supabase table 'gamme'.

    Columns expected in 'gammes' table:
      - gamme_id (text, PK or unique)
      - make (text)
      - model (text)
      - year (int)
      - trim (text)
      - type (text)  -- 'car' or 'motorcycle'
      - maintenance_schedule (jsonb)
    """

    supabase = _get_supabase_client()

    gamme_id = _build_gamme_id(vehicle)

    maintenance_json = [task.model_dump() for task in schedule]

    payload = {
        "gamme_id": gamme_id,
        "make": vehicle.make,
        "model": vehicle.model,
        "year": vehicle.year,
        "trim": vehicle.trim,
        "type": vehicle_type,
        "maintenance_schedule": maintenance_json,
    }

    # Upsert by gamme_id
    response = (
        supabase
        .table("gammes")
        .upsert(payload, on_conflict="gamme_id")
        .execute()
    )

    return {
        "gamme_id": gamme_id,
        "payload": payload,
        "response": response.data,
    }
