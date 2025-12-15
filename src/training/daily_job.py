# src/training/daily_job.py

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from src.training.incremental_trainer import train_incremental_on_new_rows
from src.data.supabase_client import get_supabase_client
from src.inference.update_gamme_params import update_gamme_params


def _yesterday_utc_iso() -> str:
    # Use UTC explicitly to avoid server timezone surprises
    today_utc = datetime.now(timezone.utc).date()
    target_date = today_utc - timedelta(days=1)
    return target_date.isoformat()


def run_daily_nnm_training() -> Dict[str, Any]:
    """
    Train the model on rows from yesterday (UTC day).
    Then update gamme parameters in DB for any gammes touched.
    """
    since_date_str = _yesterday_utc_iso()

    result = train_incremental_on_new_rows(
        since_date=since_date_str,
        epochs=1,
        batch_size=16,
        learning_rate=2e-4,     # slightly higher tends to work better with AdamW
        # if you adopted the hierarchical trainer, you likely also have:
        # weight_decay=1e-3,
        # marketcheck_weight=0.25,
        # supabase_weight=1.0,
    )

    # Only update if training actually happened
    if result.get("status") == "ok":
        for gamme_id in result.get("gamme_ids", []):
            update_gamme_params(gamme_id)

    return {
        "status": "ok",
        "since_date": since_date_str,
        "details": result,
    }


def log_training_run(payload: Dict[str, Any], error: Optional[str] = None) -> None:
    """
    Insert one row into nnm_training_runs so we can audit training.
    """
    client = get_supabase_client()

    status = "error" if error else payload.get("status", "ok")
    since_date_str = payload.get("since_date")

    details_obj = payload.get("details", {}) or {}

    insert_data = {
        "status": status,
        "since_date": since_date_str,
        "details": details_obj,  # jsonb
        "error_message": error,
    }

    resp = client.table("nnm_training_runs").insert(insert_data).execute()
    print("Logged training run to nnm_training_runs:", resp)


def main():
    """Entry point for Railway Cron."""
    print("=== Running Daily NNM Training Job ===")

    payload: Dict[str, Any] = {
        "status": "started",
        "since_date": _yesterday_utc_iso(),
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
        log_training_run(payload, error=err_msg)
        raise


if __name__ == "__main__":
    main()
