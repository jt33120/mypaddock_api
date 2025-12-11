# services/maintenance_from_receipt.py

from typing import Dict, Any, Optional, List

from openai import OpenAI

from src.config import OPENAI_API_KEY
from src.core.domain import VehicleInfo, TaskHistory
from src.infrastructure.vehicle_repository import (
    get_or_init_vehicle_history,
    update_vehicle_task_history,
)
from src.infrastructure.receipt_repository import fetch_receipt_by_id
from src.infrastructure.gamme_repository import _build_gamme_id  # optional if needed
from src.extractor.receipt_task_matcher import ReceiptTaskMatcher


MAINTENANCE_LIKE_CATEGORIES = {"Maintenance", "Repairs", "Mods"} 


def _build_vehicle_from_row(row: Dict[str, Any]) -> Optional[VehicleInfo]:
    """
    Optionnel: si tu as les infos make/model/year/trim dans vehicles,
    tu peux construire un VehicleInfo pour donner du contexte au LLM.
    Sinon tu peux retourner None.
    """
    make = row.get("make")
    model = row.get("model")
    year = row.get("year")
    trim = row.get("trim")

    if not (make and model and year):
        return None

    return VehicleInfo(
        make=make,
        model=model,
        year=year,
        trim=trim,
    )


def process_receipt_for_history(receipt_id: str) -> Dict[str, Any]:
    """
    High-level function:
    - fetch receipt
    - decide if it's relevant
    - use LLM to map to a task
    - update vehicle history
    - returns a summary dict
    """
    # 1) Fetch receipt
    receipt = fetch_receipt_by_id(receipt_id)
    if receipt is None:
        return {"updated": False, "reason": "receipt_not_found"}

    vehicle_id = receipt.get("vehicle_id")
    if not vehicle_id:
        return {"updated": False, "reason": "no_vehicle_id_on_receipt"}

    category = receipt.get("category")
    mileage = receipt.get("mileage")
    title = receipt.get("title")
    description = receipt.get("description")

    # Optional: filter by category
    if category not in MAINTENANCE_LIKE_CATEGORIES:
        return {"updated": False, "reason": "category_not_maintenance_like"}

    if not isinstance(mileage, int):
        return {"updated": False, "reason": "missing_or_invalid_mileage"}

    # 2) Get current history (or init from gamme)
    history: List[TaskHistory] = get_or_init_vehicle_history(vehicle_id)

    task_names = [h.task for h in history]

    # 3) Build receipt payload for LLM
    receipt_payload = {
        "receipt_id": receipt_id,
        "category": category,
        "title": title,
        "description": description,
        "mileage": mileage,
    }

    # 4) Optional: fetch vehicle row to give more context
    # If you have a `vehicles` repo function, you can reuse it.
    # For now, let's re-use get_or_init_vehicle_history which already pulled from vehicles.
    # We'll assume we don't have full vehicle info here -> set vehicle=None.
    vehicle_info = None  # or build from a vehicles table row if available

    # 5) LLM mapping
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Load system prompt
    from pathlib import Path
    PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "receipt_task_match_prompt.txt"
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        system_prompt = f.read()


    matcher = ReceiptTaskMatcher(client, system_prompt)
    matched_task = matcher.match_task(vehicle_info, task_names, receipt_payload)

    if matched_task is None:
        return {
            "updated": False,
            "reason": "no_task_match",
            "receipt": receipt_payload,
        }

    # 6) Update history
    updated_history = update_vehicle_task_history(
        vehicle_id=vehicle_id,
        task_name=matched_task,
        new_mileage=mileage,
    )

    return {
        "updated": True,
        "matched_task": matched_task,
        "vehicle_id": vehicle_id,
        "receipt": receipt_payload,
        "history": [h.model_dump() for h in updated_history],
    }
