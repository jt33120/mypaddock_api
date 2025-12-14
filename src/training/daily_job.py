# src/training/daily_job.py

from datetime import date, timedelta
from typing import Any, Dict, Optional

from src.training.incremental_trainer import train_incremental_on_new_rows
from src.data.supabase_client import get_supabase_client


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

    for gamme_id in result.get("gamme_ids", []):
    update_gamme_params(gamme_id)

    return {
        "status": "ok",
        "since_date": since_date_str,
        "details": result,  # this must be JSON-serializable (dict / list / primitives)
    }


def log_training_run(payload: Dict[str, Any], error: Optional[str] = None) -> None:
    """
    Insert one row into nnm_training_runs so we can audit training.

    details -> jsonb column
    """
    client = get_supabase_client()

    status = "error" if error else payload.get("status", "ok")
    since_date_str = payload.get("since_date")

    # jsonb: we pass a dict directly
    details_obj = payload.get("details", {}) or {}

    insert_data = {
        "status": status,
        "since_date": since_date_str,
        "details": details_obj,
        "error_message": error,
    }

    resp = client.table("nnm_training_runs").insert(insert_data).execute()
    print("Logged training run to nnm_training_runs:", resp)


def main():
    """Entry point for Railway Cron."""
    print("=== Running Daily NNM Training Job ===")

    # Default payload in case training fails before we get a real result
    payload: Dict[str, Any] = {
        "status": "started",
        "since_date": (date.today() - timedelta(days=1)).isoformat(),
        "details": {},
    }

    try:
        payload = run_daily_nnm_training()
        print("Training completed successfully.")
        print(payload)
        log_training_run(payload)
    except Exception as e:
        err_msg = str(e)
        print("Training failed with error:")
        print(err_msg)
        # log failed run too
        log_training_run(payload, error=err_msg)
        raise


if __name__ == "__main__":
    main()
