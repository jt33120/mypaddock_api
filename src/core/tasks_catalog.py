# core/tasks_catalog.py

from typing import List


# -----------------------------
# 🚗 CAR MAINTENANCE TASKS
# -----------------------------
CAR_TASK_CATALOG: List[str] = [
    "Engine oil and filter change",
    "Engine air filter replacement",
    "Cabin air filter replacement",
    "Spark plug replacement",
    "Brake fluid replacement",
    "Coolant replacement",
    "Transmission fluid service",
    "Brake pads inspection",
    "Tire rotation",
    "Wheel alignment check",
]


# -----------------------------
# 🏍️ MOTORCYCLE MAINTENANCE TASKS
# -----------------------------
MOTORCYCLE_TASK_CATALOG: List[str] = [
    "Engine oil and filter change",
    "Air filter cleaning or replacement",
    "Spark plug replacement",
    "Brake fluid replacement",
    "Coolant replacement",
    "Chain lubrication and tension adjustment",
    "Brake pads inspection",
    "Tire inspection and pressure check",
    "Suspension inspection",
]


def get_task_catalog(vehicle_type: str) -> List[str]:
    """
    Returns the appropriate catalog given a vehicle type string.
    vehicle_type is expected to be 'car' or 'motorcycle' (case-insensitive).
    Defaults to CAR_TASK_CATALOG if unknown.
    """
    vt = (vehicle_type or "").lower()
    if vt == "motorcycle":
        return MOTORCYCLE_TASK_CATALOG
    # default to car
    return CAR_TASK_CATALOG