from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import math
import json
import requests
import os
import re

from openai import OpenAI

from src.config import MARKETCHECK_API_KEY  # <-- define this in src/config

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@dataclass
class ValuationResult:
    price_usd: float
    comment: str
    raw_marketcheck: Optional[Dict[str, Any]] = None  # optional debug payload


class ValuatorEngine:
    """
    MarketCheck-based valuation engine.

    Primary:
      GET /v2/predict/car/us/marketcheck_price/comparables

    Fallback:
      If MarketCheck fails / returns unusable data, use OpenAI to estimate a
      private-party (used) US market value.
    """

    def __init__(
        self,
        base_url: str = "https://api.marketcheck.com",
        openai_model: str = "gpt-4o-mini",
    ) -> None:
        if not MARKETCHECK_API_KEY:
            raise RuntimeError("MARKETCHECK_API_KEY missing in .env / src.config")
        self.api_key = MARKETCHECK_API_KEY
        self.base_url = base_url.rstrip("/")

        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY missing in environment")
        self.llm = OpenAI(api_key=OPENAI_API_KEY)
        self.openai_model = openai_model

    # ---------- Internal helpers ----------

    def _safe_get(self, data: Dict[str, Any], key: str) -> str:
        """Convert None / NaN to 'unknown'."""
        v = data.get(key, None)
        if v is None:
            return "unknown"
        if isinstance(v, float) and math.isnan(v):
            return "unknown"
        return str(v)

    # Supabase schema: vin/mileage/zip
    def _extract_vin(self, vehicle: Dict[str, Any]) -> Optional[str]:
        vin = vehicle.get("vin")
        if not vin:
            return None
        return str(vin).strip()

    def _extract_miles(self, vehicle: Dict[str, Any]) -> Optional[int]:
        mileage = vehicle.get("mileage")
        if mileage is None:
            return None
        try:
            miles = int(mileage)
        except (TypeError, ValueError):
            return None
        return miles if miles >= 0 else None

    def _extract_zip(self, vehicle: Dict[str, Any]) -> Optional[str]:
        z = vehicle.get("zip")
        if not z:
            return None
        return str(z).strip()

    def _extract_dealer_type(self, vehicle: Dict[str, Any]) -> str:
        """
        MarketCheck requires dealer_type. Your prior code used "independent".
        If you're valuing used/private-party-ish listings, "used" is fine
        if MarketCheck accepts it. If you see MC errors, switch back to "independent".
        """
        v = vehicle.get("dealer_type")
        if isinstance(v, str) and v.strip():
            return v.strip()
        return "independent"  # closest to original MarketCheck-compatible default

    def _call_marketcheck_price(
        self,
        vin: str,
        miles: int,
        dealer_type: str,
        zip_code: str,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/v2/predict/car/us/marketcheck_price/comparables"
        params = {
            "api_key": self.api_key,
            "vin": vin,
            "miles": miles,
            "dealer_type": dealer_type,
            "zip": zip_code,
        }

        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            raise requests.HTTPError(
                f"MarketCheck returned {resp.status_code}: {resp.text}",
                response=resp,
            )
        return resp.json()

    def _is_mc_data_usable(self, mc_data: Dict[str, Any]) -> bool:
        if not isinstance(mc_data, dict):
            return False
        price = mc_data.get("marketcheck_price")
        try:
            price_f = float(price)
        except (TypeError, ValueError):
            return False
        return price_f > 0

    def _extract_json_object(self, text: str) -> Dict[str, Any]:
        """
        Defensive fallback if the model includes extra text.
        Tries to find the first {...} JSON object in the response.
        """
        text = text.strip()
        # Fast path: pure JSON
        try:
            return json.loads(text)
        except Exception:
            pass

        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise ValueError("No JSON object found in LLM output")
        return json.loads(m.group(0))

    def _llm_fallback_valuation(self, vehicle: Dict[str, Any], reason: str) -> ValuationResult:
        """
        Old-logic-style fallback prompt. Always returns ValuationResult.
        If the LLM fails, returns price_usd=0.0 with an error comment.
        """
        prompt = f"""
Estimate TODAY'S value of the following vehicle in USD (United States, private-party/used).

IMPORTANT RULES:
- If information is missing or 'unknown', explain its impact in the comment.
- Consider mileage, age, trim, and condition logically.
- Output valid JSON only. Do NOT add commentary outside JSON.
- Use US dollars for price.

Vehicle data:
{json.dumps(vehicle, ensure_ascii=False, indent=2)}

Reason for fallback:
{reason}

Return JSON exactly in this format:
{{
  "price_usd": 12345,
  "comment": "string describing confidence and missing fields"
}}
""".strip()

        try:
            resp = self.llm.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You estimate used vehicle values conservatively and realistically."},
                    {"role": "user", "content": prompt},
                ],
                # Key improvement vs your current version: enforce JSON output
                response_format={"type": "json_object"},
                temperature=0.2,
            )

            content = resp.choices[0].message.content or ""
            parsed = self._extract_json_object(content)

            price = float(parsed.get("price_usd", 0.0))
            comment = str(parsed.get("comment", "")).strip() or "LLM fallback valuation."

            if price <= 0:
                # Old behavior was "shouldn't be 0"; if it is, treat as failure
                return ValuationResult(
                    price_usd=0.0,
                    comment=f"LLM fallback returned non-positive price. Comment: {comment}",
                    raw_marketcheck=None,
                )

            return ValuationResult(
                price_usd=price,
                comment=f"[LLM fallback] {comment}",
                raw_marketcheck=None,
            )

        except Exception as e:
            return ValuationResult(
                price_usd=0.0,
                comment=f"LLM fallback failed: {e}",
                raw_marketcheck=None,
            )

    # ---------- Public API ----------

    def evaluate(self, vehicle: Dict[str, Any]) -> ValuationResult:
        """
        Returns ValuationResult. Never returns None.
        """
        if vehicle is None:
            return ValuationResult(
                price_usd=0.0,
                comment="Vehicle not found in database.",
                raw_marketcheck=None,
            )

        vin = self._extract_vin(vehicle)
        miles = self._extract_miles(vehicle)
        zip_code = self._extract_zip(vehicle)
        dealer_type = self._extract_dealer_type(vehicle)

        missing_fields = []
        if not vin:
            missing_fields.append("vin")
        if miles is None:
            missing_fields.append("mileage")
        if not zip_code:
            missing_fields.append("zip")

        if missing_fields:
            return self._llm_fallback_valuation(
                vehicle,
                reason=f"Missing required fields for MarketCheck: {', '.join(missing_fields)}",
            )

        try:
            mc_data = self._call_marketcheck_price(
                vin=vin,
                miles=miles,
                dealer_type=dealer_type,
                zip_code=zip_code,
            )
        except Exception as e:
            return self._llm_fallback_valuation(
                vehicle,
                reason=f"MarketCheck Price API call failed: {e}",
            )

        if not self._is_mc_data_usable(mc_data):
            return self._llm_fallback_valuation(
                vehicle,
                reason="MarketCheck returned missing/invalid/zero marketcheck_price",
            )

        price_float = float(mc_data.get("marketcheck_price"))

        msrp = mc_data.get("msrp")
        comps = mc_data.get("comparables", {}) or {}
        num_comps = comps.get("num_found") or len(comps.get("listings", []) or [])

        comment_parts = [
            f"Price from MarketCheck for VIN {vin}.",
            f"Miles: {miles}, zip: {zip_code}, dealer_type: {dealer_type}.",
            f"Comparables used: {num_comps}.",
        ]
        if msrp is not None:
            comment_parts.append(f"MSRP: {msrp} USD.")

        return ValuationResult(
            price_usd=price_float,
            comment=" ".join(comment_parts),
            raw_marketcheck=mc_data,
        )
