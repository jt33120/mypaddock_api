# extractor/receipt_task_matcher.py

import json
from typing import List, Optional, Any, Dict
from openai import OpenAI

from core.domain import VehicleInfo


def _extract_json_from_text(text: str) -> Any:
    """
    Extract JSON from model output, handling ```json fences.
    """
    text = text.strip()

    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    return json.loads(text)


class ReceiptTaskMatcher:
    """
    Uses OpenAI to map a receipt to a maintenance task name (or None).
    """

    def __init__(self, client: OpenAI, system_prompt: str):
        self.client = client
        self.system_prompt = system_prompt

    def match_task(
        self,
        vehicle: Optional[VehicleInfo],
        task_names: List[str],
        receipt: Dict[str, Any],
    ) -> Optional[str]:
        """
        Given a vehicle (optional), a list of canonical task names,
        and a receipt dict {category, title, description, mileage, ...},
        returns the matched task name or None.
        """
        vehicle_payload = vehicle.model_dump() if vehicle is not None else None

        user_payload = {
            "vehicle": vehicle_payload,
            "tasks": task_names,
            "receipt": receipt,
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
        data = _extract_json_from_text(raw_text)

        matched = data.get("matched_task", None)

        if matched is None:
            return None

        # Safety: ensure it's in the list, otherwise ignore
        if matched not in task_names:
            return None

        return matched
