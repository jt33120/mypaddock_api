# src/training/daily_job.py

from datetime import date, timedelta
from typing import Any, Dict

from src.training.incremental_trainer import train_incremental_on_new_rows


def run_daily_nnm_training() -> Dict[str, Any]:
    """
    Train the NNM on all rows from yesterday in vehicle_timeseries.

    We use yesterday because the job runs after midnight (e.g. UTC 10:00),
    so "yesterday" refers to the complete day in all US time zones.
    """
    target_date = date.today() - timedelta(days=1)
    since_date_str = target_date.isoformat()

    result = train_incremental_on_new_rows(
        since_date=since_date_str,
        epochs=1,
        batch_size=16,
        learning_rate=1e-4,
    )

    return {
        "status": "ok",
        "since_date": since_date_str,
        "details": result,
    }


def main():
    """Entry point for Railway Cron."""
    print("=== Running Daily NNM Training Job ===")

    try:
        result = run_daily_nnm_training()
        print("Training completed successfully.")
        print(result)
    except Exception as e:
        print("Training failed with error:")
        print(str(e))
        raise


if __name__ == "__main__":
    main()
