from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import math
import json
import requests
import os

from openai import OpenAI

from src.data.preprocessing import VEHICLE_FEATURES
from src.config import MARKETCHECK_API_KEY  # <-- define this in src/config


# Read from env (recommended). You can also move this into src.config if you prefer.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@dataclass
class ValuationResult:
    price_usd: float
    comment: str
    raw_marketcheck: Optional[Dict[str, Any]] = None  # optional debug payload


class ValuatorEngine:
    """
    MarketCheck-based valuation engine.

    Uses MarketCheck's MarketCheck Price (comparables) endpoint to estimate
    the vehicle's value:

      GET /v2/predict/car/us/marketcheck_price/comparables

    Fallback:
      If MarketCheck fails / returns 0 / returns missing info, uses an OpenAI prompt
      to estimate a private-party US market value.
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
        """Convert None / NaN to 'unknown' (kept in case you still use VEHICLE_FEATURES somewhere)."""
        v = data.get(key, None)
        if v is None:
            return "unknown"
        if isinstance(v, float) and math.isnan(v):
            return "unknown"
        return str(v)

    def _extract_vin(self, vehicle: Dict[str, Any]) -> Optional[str]:
        vin = vehicle.get("vin")
        if vin:
            return str(vin)
        return None


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
        zip_code = vehicle.get("zip")
        if zip_code:
            return str(zip_code)
        return None


    def _extract_dealer_type(self, vehicle: Dict[str, Any]) -> str:
        """
        MarketCheck requires dealer_type, but for private / 3rd-party valuations
        we can default to 'independent' if we don't have better info.
        """
        v = vehicle.get("dealer_type")
        if isinstance(v, str) and v:
            return v
        # Default assumption for MyPaddock: used 
        return "used"

    def _call_marketcheck_price(
        self,
        vin: str,
        miles: int,
        dealer_type: str,
        zip_code: str,
    ) -> Dict[str, Any]:
        """
        Perform the HTTP call to MarketCheck.

        Raises requests.HTTPError if status != 200.
        """
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
        """
        Decide whether MarketCheck response contains enough info to trust.
        We require a positive numeric marketcheck_price.
        """
        if not isinstance(mc_data, dict):
            return False

        price = mc_data.get("marketcheck_price")
        try:
            price_f = float(price)
        except (TypeError, ValueError):
            return False

        if price_f <= 0:
            return False

        return True

    def _llm_fallback_valuation(self, vehicle: Dict[str, Any], reason: str) -> Optional[ValuationResult]:
        """
        Simple LLM-based fallback valuation (private-party, US).
        Returns ONLY if MC fails / returns unusable data.
        """
        prompt = f"""
You are an automotive valuation expert.

Estimate a fair private-party market value in USD for the vehicle below.
Return ONLY valid JSON with:
- price_usd (number)
- explanation (string)

Vehicle data:
{json.dumps(vehicle, ensure_ascii=False, indent=2)}

Context:
- Sale type: used
- Market: United States
"""

        try:
            resp = self.llm.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You estimate used car values conservatively and realistically."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )

            content = resp.choices[0].message.content or ""
            parsed = json.loads(content)

            price = float(parsed.get("price_usd", 0.0))
            explanation = str(parsed.get("explanation", "LLM fallback valuation."))

            if price <= 0:
                return None

            return ValuationResult(
                price_usd=price,
                comment=f"[LLM fallback] {explanation}",
                raw_marketcheck=None,
            )

        except Exception as e:
            return ValuationResult(
                price_usd=None,
                comment=f"LLM fallback failed: {e}",
                raw_marketcheck=None,
            )

    # ---------- Public API ----------

    def evaluate(self, vehicle: Dict[str, Any]) -> ValuationResult:
        """
        Main entrypoint: given a vehicle dict from your DB, return a ValuationResult.

        Required for MarketCheck:
          - vin
          - mileage (miles)
          - dealer_type (we default to 'independent' if missing)
          - zip (location)

        Fallback to LLM if:
          - required fields are missing
          - MarketCheck call errors
          - MarketCheck returns 0 / invalid / missing marketcheck_price
        """

        if vehicle is None:
            # You can keep this as a hard zero or also fallback — your choice.
            return ValuationResult(
                price_usd=None,
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

        # If response is missing / invalid price, fallback
        if not self._is_mc_data_usable(mc_data):
            return self._llm_fallback_valuation(
                vehicle,
                reason="MarketCheck returned missing/invalid/zero marketcheck_price",
            )

        # Parse response
        price = mc_data.get("marketcheck_price")
        msrp = mc_data.get("msrp")
        comps = mc_data.get("comparables", {}) or {}
        num_comps = comps.get("num_found") or len(comps.get("listings", []) or [])

        price_float = float(price)

        comment_parts = [
            f"Price from MarketCheck Price Premium for VIN {vin}.",
            f"Miles: {miles}, zip: {zip_code}, dealer_type: {dealer_type}.",
            f"Comparables used: {num_comps}.",
        ]
        if msrp is not None:
            comment_parts.append(f"MSRP: {msrp} USD.")

        comment = " ".join(comment_parts)

        return ValuationResult(
            price_usd=price_float,
            comment=comment,
            raw_marketcheck=mc_data,
        )
