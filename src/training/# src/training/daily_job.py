# src/training/daily_job.py
from datetime import date
from typing import Any, Dict
from src.training.incremental_trainer import train_incremental_on_new_rows

def run_daily_nnm_training() -> Dict[str, Any]:
    """
    Train the NNM on all rows from today in vehicle_timeseries.
    This is meant to be called once per day (e.g. by a cron).
    """
    today_str = (date.today() - timedelta(days=1)).isoformat()

    result = train_incremental_on_new_rows(
        since_date=today_str,
        epochs=1,
        batch_size=16,
        learning_rate=1e-4,
    )

    # Optionally add metadata
    return {
        "status": "ok",
        "since_date": today_str,
        "details": result,
    }
