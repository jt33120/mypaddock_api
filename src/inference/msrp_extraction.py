from __future__ import annotations

import json
from typing import Optional, Dict, Any

from openai import OpenAI

from src.config import OPENAI_API_KEY
from src.data.supabase_client import get_supabase_client


TABLE_NAME = "vehicles_market_parameters"


def _build_msrp_prompt(
    make: str,
    model: str,
    year: int,
    trim: Optional[str],
    country: str,
) -> str:
    trim_str = trim if trim else "base / standard trim"

    return f"""
You are an automotive pricing expert.

Task:
Estimate the **MSRP (Manufacturer's Suggested Retail Price)** in **USD** for the following car
at the time it was **brand new**, in the specified country.

Car:
- Make: {make}
- Model: {model}
- Model year: {year}
- Trim: {trim_str}
- Country: {country}

Important rules:
- Return a **single numeric MSRP in USD** (no range).
- If the car was originally priced in another currency, convert to USD approximately.
- If multiple engine/options existed for this trim, pick the most common MSRP.
- If you're not sure, give your best estimate with a lower confidence.

Return ONLY valid JSON with this exact shape:

{{
  "msrp_usd": <number>,
  "reference_year": <number>,  // the year this MSRP refers to (usually the model year)
  "confidence": "<high|medium|low>",
  "note": "<very short explanation>"
}}
"""


def fetch_msrp_from_openai(
    make: str,
    model: str,
    year: int,
    trim: Optional[str],
    country: str,
) -> Dict[str, Any]:
    """
    Call OpenAI to get an MSRP estimate in USD.

    Returns dict with keys:
    - msrp_usd: float or None
    - reference_year: int or None
    - confidence: str
    - note: str
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing in config / .env")

    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = _build_msrp_prompt(make, model, year, trim, country)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a precise automotive pricing model. Return ONLY JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )

    content = response.choices[0].message.content

    try:
        obj = json.loads(content)
        msrp_usd = float(obj["msrp_usd"])
        ref_year = int(obj.get("reference_year", year))
        confidence = obj.get("confidence", "unknown")
        note = obj.get("note", "")
        return {
            "msrp_usd": msrp_usd,
            "reference_year": ref_year,
            "confidence": confidence,
            "note": note,
        }
    except Exception as e:
        return {
            "msrp_usd": None,
            "reference_year": None,
            "confidence": "error",
            "note": f"Failed to parse JSON: {e}. Raw content: {content}",
        }


def upsert_msrp_for_market_car(
    make: str,
    model: str,
    year: int,
    trim: Optional[str],
    country: str,
) -> Dict[str, Any]:
    """
    High-level helper:
    - Look for a row in vehicles_market_nn_parameters matching (make, model, year, trim, country)
    - If not found, create it.
    - Call OpenAI to get MSRP in USD.
    - Update V0 in the row (overwrite if already present).

    Returns a dict with status + row data.
    """
    client = get_supabase_client()

    # 1) Fetch or create the Market Car row
    query = (
        client.table(TABLE_NAME)
        .select("*")
        .eq("make", make)
        .eq("model", model)
        .eq("year", year)
        .eq("country", country)
    )

    if trim is not None:
        query = query.eq("trim", trim)
    else:
        query = query.is_("trim", None)

    resp = query.limit(1).execute()
    rows = resp.data or []

    if rows:
        row = rows[0]
        row_id = row["vehicle_market_id"]
    else:
        # Create new row
        payload = {
            "make": make,
            "model": model,
            "year": year,
            "trim": trim,
            "country": country,
        }
        insert_resp = client.table(TABLE_NAME).insert(payload).execute()
        row = insert_resp.data[0]
        row_id = row["vehicle_market_id"]

    # 2) Ask OpenAI for MSRP
    msrp_info = fetch_msrp_from_openai(make, model, year, trim, country)
    msrp = msrp_info["msrp_usd"]

    if msrp is None:
        return {
            "status": "error",
            "vehicle_market_id": row_id,
            "msrp_info": msrp_info,
        }

    # 3) Update the row's V0 (MSRP) and optionally store reference_year
    update_payload = {
        "V0": msrp,
        # If you later add columns like `msrp_reference_year`, `msrp_confidence`, etc.,
        # you can update them here too, e.g.:
        # "msrp_reference_year": msrp_info["reference_year"],
        # "msrp_confidence": msrp_info["confidence"],
    }

    update_resp = (
        client.table(TABLE_NAME)
        .update(update_payload)
        .eq("vehicle_market_id", row_id)
        .execute()
    )

    updated_row = update_resp.data[0] if update_resp.data else row

    return {
        "status": "ok",
        "vehicle_market_id": row_id,
        "msrp_usd": msrp,
        "msrp_info": msrp_info,
        "row": updated_row,
    }
