from __future__ import annotations

from fastapi import FastAPI

from src.api.schemas import RefreshRequest, RefreshResponse, VehicleValuation
from src.inference.refresh import refresh_user_vehicles

app = FastAPI(title="MyPaddock Valuation API")


@app.post("/valuation/refresh", response_model=RefreshResponse)
async def refresh_valuation(req: RefreshRequest) -> RefreshResponse:
    """
    Refresh valuations for all vehicles owned by the given user.

    - For each vehicle, only one valuation per day is computed.
    - If a valuation exists today, it is not recomputed.
    - Uses the current ValuatorEngine implementation (LLM-based for now).
    """
    raw_results = refresh_user_vehicles(req.user_id)

    valuations = [VehicleValuation(**r) for r in raw_results]

    return RefreshResponse(
        updated=len(valuations),
        valuations=valuations,
    )
