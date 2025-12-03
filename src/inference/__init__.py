from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.inference.refresh import refresh_user_vehicles
from src.inference.new_gamme import update_gamme_row_from_llm 


# 1) Create FastAPI app ONCE
app = FastAPI()

origins = [
    "http://localhost:3000",
    # add your deployed frontend URL here, for example:
    # "https://mypaddock.vercel.app",
]

# 2) Add CORS middleware (so your React app can call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # later you can restrict to ["http://localhost:5173"]
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 3) Request/Response models for the endpoint
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
    trim: Optional[str] = ""  # keep it simple, you can adjust



# 4) Endpoint that uses your refresh_user_vehicles() function
@app.post("/valuation/refresh", response_model=List[RefreshResponse])
def valuation_refresh(req: RefreshRequest):
    """
    Trigger valuations refresh for all vehicles of a user.
    Reuses refresh_user_vehicles from refresh.py
    """
    if not req.user_id:
        raise HTTPException(status_code=400, detail="Missing user_id")

    results = refresh_user_vehicles(req.user_id)
    return results

@app.post("/valuation/new_gamme")
def valuation_new_gamme(req: NewGammeRequest):
    """
    Create / update a gamme row via LLM and Supabase.
    This is what your React frontend calls.
    """
    if not req.make or not req.model or not req.year:
        raise HTTPException(status_code=400, detail="Missing make/model/year")

    try:
        payload = update_gamme_row_from_llm(
            make=req.make,
            model=req.model,
            year=req.year,
            trim=req.trim or "",
        )
        # You can choose what you return; here we return the whole payload for debugging
        return payload
    except Exception as e:
        # log e on server side if you want more detail
        raise HTTPException(status_code=500, detail=str(e))
