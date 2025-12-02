from __future__ import annotations

from typing import List
from pydantic import BaseModel


class RefreshRequest(BaseModel):
    user_id: str


class VehicleValuation(BaseModel):
    vehicle_id: str
    price_usd: float
    comment: str


class RefreshResponse(BaseModel):
    updated: int
    valuations: List[VehicleValuation]
