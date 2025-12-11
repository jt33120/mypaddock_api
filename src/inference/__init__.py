from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.inference.refresh import refresh_user_vehicles
from src.inference.new_gamme import update_gamme_row_from_llm

# ✅ NEW IMPORTS
from src.infrastructure.vehicle_repository import get_or_init_vehicle_history
from src.service.maintenance_from_receipt import process_receipt_for_history


# 1) Create FastAPI app ONCE
app = FastAPI()

# CORS
origins = [
    "http://localhost:3000",
    # "https://mypaddock.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] if you want completely open CORS
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 2) Request/Response models
class RefreshRequest(BaseModel):
    user_id: str


class RefreshResponse(BaseModel):
    vehicle_id: str
    price_usd: float
    comment: Optional[str] = None


class NewGammeRequest(BaseModel):
    make: str
    model: str
    year: int
    trim: Optional[str] = ""


# ✅ NEW REQUEST MODELS FOR MAINTENANCE
class VehicleHistoryRequest(BaseModel):
    vehicle_id: str


class ReceiptHistoryRequest(BaseModel):
    receipt_id: str


# 3) Endpoints

@app.post("/valuation/refresh", response_model=List[RefreshResponse])
def valuation_refresh(req: RefreshRequest):
    """
    Trigger valuations refresh for all vehicles of a user.
    """
    if not req.user_id:
        raise HTTPException(status_code=400, detail="Missing user_id")

    return refresh_user_vehicles(req.user_id)


@app.post("/valuation/new_gamme")
def valuation_new_gamme(req: NewGammeRequest):
    """
    Create / update a gamme row via LLM and Supabase.
    """
    if not req.make or not req.model or not req.year:
        raise HTTPException(status_code=400, detail="Missing make/model/year")

    try:
        return update_gamme_row_from_llm(
            make=req.make,
            model=req.model,
            year=req.year,
            trim=req.trim or "",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ✅ NEW ENDPOINT: init / get vehicle history
@app.post("/maintenance/init-history")
def maintenance_init_history(req: VehicleHistoryRequest):
    """
    Ensure the vehicle history exists for this vehicle (idempotent).
    """
    if not req.vehicle_id:
        raise HTTPException(status_code=400, detail="Missing vehicle_id")

    try:
        result = get_or_init_vehicle_history(req.vehicle_id)
        return result
    except Exception as e:
        # Log or handle as you wish
        raise HTTPException(status_code=500, detail=str(e))


# ✅ NEW ENDPOINT: process a receipt into history
@app.post("/maintenance/process-receipt")
def maintenance_process_receipt(req: ReceiptHistoryRequest):
    """
    Process a given receipt into the vehicle's maintenance history.
    """
    if not req.receipt_id:
        raise HTTPException(status_code=400, detail="Missing receipt_id")

