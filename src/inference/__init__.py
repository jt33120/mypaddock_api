from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.inference.refresh import refresh_user_vehicles


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
