from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import math
import json
import requests

from src.data.preprocessing import VEHICLE_FEATURES
from src.config import MARKETCHECK_API_KEY  # <-- you'll define this in src/config


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

    It returns:
      - estimated price in USD (marketcheck_price)
      - a short comment about what was used / assumed
    """

    def __init__(
        self,
        base_url: str = "https://api.marketcheck.com",
    ) -> None:
        if not MARKETCHECK_API_KEY:
            raise RuntimeError("MARKETCHECK_API_KEY missing in .env / src.config")
        self.api_key = MARKETCHECK_API_KEY
        self.base_url = base_url.rstrip("/")

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
        for k in ("vin", "VIN", "vehicle_vin"):
            if k in vehicle and vehicle[k]:
                return str(vehicle[k])
        return None

    def _extract_miles(self, vehicle: Dict[str, Any]) -> Optional[int]:
        for k in ("mileage", "miles", "odometer", "odometer_miles"):
            v = vehicle.get(k)
            if v is None:
                continue
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
        return None

    def _extract_zip(self, vehicle: Dict[str, Any]) -> Optional[str]:
        # Try various potential keys: adjust as needed for your schema
        for k in ("zip", "zip_code", "postal_code", "owner_zip", "owner_postal_code"):
            v = vehicle.get(k)
            if v:
                return str(v)
        return None

    def _extract_dealer_type(self, vehicle: Dict[str, Any]) -> str:
        """
        MarketCheck requires dealer_type, but for private / 3rd-party valuations
        we can default to 'independent' if we don't have better info.
        """
        v = vehicle.get("dealer_type")
        if isinstance(v, str) and v:
            return v
        # Default assumption for MyPaddock: private or independent dealers
        return "independent"

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
            # Let caller handle the error
            raise requests.HTTPError(
                f"MarketCheck returned {resp.status_code}: {resp.text}",
                response=resp,
            )
        return resp.json()

    # ---------- Public API ----------

    def evaluate(self, vehicle: Dict[str, Any]) -> ValuationResult:
        """
        Main entrypoint: given a vehicle dict from your DB, return a ValuationResult.

        Required for MarketCheck:
          - vin
          - mileage (miles)
          - dealer_type (we default to 'independent' if missing)
          - zip (location)
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
            return ValuationResult(
                price_usd=0.0,
                comment=(
                    "Cannot call MarketCheck: missing required fields: "
                    + ", ".join(missing_fields)
                ),
                raw_marketcheck=None,
            )

        try:
            mc_data = self._call_marketcheck_price(
                vin=vin,
                miles=miles,
                dealer_type=dealer_type,
                zip_code=zip_code,
            )
        except Exception as e:
            return ValuationResult(
                price_usd=0.0,
                comment=f"MarketCheck Price API call failed: {e}",
                raw_marketcheck=None,
            )

        # Parse response
        price = mc_data.get("marketcheck_price")
        msrp = mc_data.get("msrp")
        comps = mc_data.get("comparables", {}) or {}
        num_comps = comps.get("num_found") or len(comps.get("listings", []) or [])

        try:
            price_float = float(price) if price is not None else 0.0
        except (TypeError, ValueError):
            price_float = 0.0

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
