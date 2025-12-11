# infrastructure/vehicle_repository.py

from typing import Optional, List, Dict, Any

from supabase import create_client, Client

from src.core.domain import VehicleInfo, TaskHistory, TaskHistoryList
from config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY


def _get_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Supabase URL or SERVICE_ROLE key not set in config.")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def _fetch_vehicle_row(client: Client, vehicle_id: str) -> Optional[Dict[str, Any]]:
    resp = (
        client.table("vehicles")
        .select("*")
        .eq("vehicle_id", vehicle_id)
        .single()
        .execute()
    )
    return resp.data  # None si pas trouvé


def _fetch_gamme_schedule(client: Client, gamme_id: str) -> List[Dict[str, Any]]:
    """
    Récupère la liste de tasks depuis la table 'gamme'.maintenance_schedule.
    On suppose que maintenance_schedule est un JSON array de { task, interval }.
    """
    resp = (
        client.table("gammes")
        .select("maintenance_schedule")
        .eq("gamme_id", gamme_id)
        .single()
        .execute()
    )
    data = resp.data or {}
    return data.get("maintenance_schedule") or []


def _build_initial_history_from_schedule(schedule: List[Dict[str, Any]]) -> TaskHistoryList:
    """
    À partir du schedule de la gamme, on crée l'historique initial :
    chaque task a last_mileage = None.
    """
    history: TaskHistoryList = []

    for item in schedule:
        task_name = item.get("task")
        if not task_name:
            continue
        history.append(TaskHistory(task=task_name, last_mileage=None))

    return history


def get_or_init_vehicle_history(vehicle_id: str) -> TaskHistoryList:
    """
    Pour un vehicle_id :
      - Si history existe déjà dans la table vehicles, on le renvoie (en TaskHistoryList).
      - Sinon :
          - On récupère la gamme correspondante (via gamme_id).
          - On construit un history initial à partir du maintenance_schedule de la gamme.
          - On enregistre cet history dans vehicles.history.
          - On renvoie l'history.
    """
    client = _get_supabase_client()

    vehicle_row = _fetch_vehicle_row(client, vehicle_id)
    if vehicle_row is None:
        raise ValueError(f"Vehicle with id {vehicle_id} not found in 'vehicles' table.")

    existing_history = vehicle_row.get("history")
    if existing_history:
        # On reconstruit des TaskHistory à partir du JSON
        return [TaskHistory(**h) for h in existing_history]

    # Sinon, il faut initialiser à partir de la gamme
    gamme_id = vehicle_row.get("gamme_id")
    if not gamme_id:
        raise ValueError(f"Vehicle {vehicle_id} has no gamme_id defined.")

    schedule = _fetch_gamme_schedule(client, gamme_id)
    history = _build_initial_history_from_schedule(schedule)

    # Sauvegarde dans la table vehicles
    history_json = [h.model_dump() for h in history]

    (
        client.table("vehicles")
        .update({"history": history_json})
        .eq("vehicle_id", vehicle_id)
        .execute()
    )

    return history


def update_vehicle_task_history(
    vehicle_id: str,
    task_name: str,
    new_mileage: int,
) -> TaskHistoryList:
    """
    Met à jour l'historique d'un véhicule pour UNE tâche donnée :
      - si la tâche existe dans history, on met à jour last_mileage.
      - sinon, on l'ajoute (au cas où une nouvelle task apparaît un jour).
    Sauvegarde l'history dans Supabase et renvoie la liste mise à jour.
    """
    client = _get_supabase_client()

    # On récupère l'historique actuel (ou on l'initialise si absent)
    history = get_or_init_vehicle_history(vehicle_id)

    # Mise à jour en mémoire
    updated = False
    for entry in history:
        if entry.task == task_name:
            entry.last_mileage = new_mileage
            updated = True
            break

    if not updated:
        # Task inconnue dans l'historique → on l'ajoute
        history.append(TaskHistory(task=task_name, last_mileage=new_mileage))

    # Sauvegarde en base
    history_json = [h.model_dump() for h in history]

    (
        client.table("vehicles")
        .update({"history": history_json})
        .eq("vehicle_id", vehicle_id)
        .execute()
    )

    return history
