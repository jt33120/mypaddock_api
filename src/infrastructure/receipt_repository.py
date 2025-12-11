# infrastructure/receipt_repository.py

from typing import Optional, Dict, Any

from supabase import create_client, Client

from src.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY


def _get_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Supabase URL or SERVICE_ROLE key not set in config.")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def fetch_receipt_by_id(receipt_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetches a receipt row from the 'receipts' table by receipt_id.
    """
    client = _get_supabase_client()

    resp = (
        client.table("receipts")
        .select("*")
        .eq("receipt_id", receipt_id)
        .single()
        .execute()
    )

    return resp.data
