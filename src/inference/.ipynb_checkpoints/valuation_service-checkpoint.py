from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import math
import json

from src.data.preprocessing import VEHICLE_FEATURES
from src.config import OPENAI_API_KEY

from openai import OpenAI


@dataclass
class ValuationResult:
    price_usd: float
    comment: str


class ValuatorEngine:
    """
    LLM-based valuation engine.
    Uses OpenAI to generate a structured valuation:
    - estimated price in USD
    - comment about missing information
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY missing in .env")
        # If OPENAI_API_KEY is set in env, passing it here is optional, but explicit is fine
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model

    def _safe_get(self, data: Dict[str, Any], key: str) -> str:
        """Convert None / NaN to 'unknown'."""
        v = data.get(key, None)
        if v is None:
            return "unknown"
        if isinstance(v, float) and math.isnan(v):
            return "unknown"
        return str(v)

    def _build_prompt(self, vehicle: Dict[str, Any]) -> str:
        """
        Creates the prompt sent to OpenAI.
        We instruct the model to return VALID JSON.
        """

        vehicle_info = {
            feat: self._safe_get(vehicle, feat)
            for feat in VEHICLE_FEATURES
        }

        return f"""
You are a vehicle valuation expert.
Estimate TODAY'S value of the following vehicle in USD.

IMPORTANT RULES:
- If information is missing or 'unknown', explain its impact.
- Consider mileage, age, trim, and condition logically.
- Output valid JSON only. Do NOT add commentary outside JSON.
- Use US dollars for price.

Vehicle data:
{json.dumps(vehicle_info, indent=2)}

Return JSON in this format exactly:

{{
  "price_usd": <number>,
  "comment": "<string describing confidence and missing fields>"
}}
"""

    def evaluate(self, vehicle: Dict[str, Any]) -> ValuationResult:
        if vehicle is None:
            return ValuationResult(
                price_usd=0.0,
                comment="Vehicle not found in database."
            )

        prompt = self._build_prompt(vehicle)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a precise valuation model."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        # ✅ New SDK: content is an attribute, not a dict
        content = response.choices[0].message.content

        # Parse JSON
        try:
            data = json.loads(content)
            price = float(data.get("price_usd", 0))
            comment = data.get("comment", "No comment provided.")
        except Exception as e:
            # Fallback if JSON failed
            price = 0.0
            comment = f"Failed to parse model JSON: {e}. Raw content: {content}"

        return ValuationResult(price_usd=price, comment=comment)
