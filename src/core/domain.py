# core/domain.py

from pydantic import BaseModel
from typing import Optional, List


class MaintenanceTask(BaseModel):
    task: str
    interval: Optional[int] = None  # one interval value, miles or years


class VehicleInfo(BaseModel):
    make: str
    model: str
    year: int
    trim: Optional[str] = None


class TaskHistory(BaseModel):
    task: str
    last_mileage: Optional[int] = None  # in miles, None if never done


TaskHistoryList = List[TaskHistory]
