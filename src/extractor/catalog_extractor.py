# extractor/catalog_extractor.py

import json
from typing import List, Any
from openai import OpenAI

from core.domain import VehicleInfo, MaintenanceTask
from core.tasks_catalog import get_task_catalog


def _extract_json_from_text(text: str) -> Any:
    """
    Extract a JSON array from model output.
    Handles ```json fences.
    """
    text = text.strip()

    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    return json.loads(text)


class CatalogFrequencyExtractor:
    """
    V1 extractor:
    - Uses a generic task catalog (car or motorcycle),
    - Lets the LLM assign ONE interval per task (miles OR years),
    - Returns a simplified schedule { task, interval }.
    """

    def __init__(self, client: OpenAI, system_prompt: str):
        self.client = client
        self.system_prompt = system_prompt

    def generate_schedule(self, vehicle: VehicleInfo, vehicle_type: str) -> List[MaintenanceTask]:
        task_names = get_task_catalog(vehicle_type)

        base_items = [
            { "task": task, "interval": None }
            for task in task_names
        ]

        user_payload = {
            "vehicle": vehicle.model_dump(),
            "vehicle_type": vehicle_type,
            "tasks": base_items,
        }

        completion = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            temperature=0,
        )

        raw_text = completion.choices[0].message.content
        raw = _extract_json_from_text(raw_text)

        schedule: List[MaintenanceTask] = []

        if not isinstance(raw, list):
            raise ValueError("Expected JSON array from model")

        for item in raw:
            task_name = item.get("task")
            interval = item.get("interval")

            # ensure interval is int or None
            if not isinstance(interval, int):
                interval = None

            schedule.append(
                MaintenanceTask(task=task_name, interval=interval)
            )

        return schedule
