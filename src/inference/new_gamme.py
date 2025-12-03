# src/inference/new_gamme.py
from __future__ import annotations

import json
from typing import Dict, Any

from openai import OpenAI
from src.data.supabase_client import get_supabase_client

# Adjust model name if you use something else
OPENAI_MODEL = "gpt-4.1-mini"

openai_client = OpenAI()


def _build_gamme_id(make: str, model: str, year: int | str, trim: str) -> str:
    """
    Build gamme_id as MAKEMODELTRIMYEAR, uppercased and without spaces.
    Example: BMWM3COMPETITION2018
    """
    return f"{make}{model}{trim}{year}".replace(" ", "").upper()


def _fetch_gamme_specs_from_llm(
    make: str,
    model: str,
    year: int | str,
    trim: str,
) -> Dict[str, Any]:
    """
    Call OpenAI to get:
      - msrp_usd_v0
      - engine_displacement (L or cc, we normalize to liters)
      - hp
      - engine_configuration (e.g. inline, V, flat)
      - number_of_cylinders
      - supply_by_country (JSON object, best-effort)
    """

    messages = [
        {
            "role": "system",
            "content": (
                "You are an automotive data assistant. "
                "Given make, model, year and trim, you estimate technical specs and MSRP. "
                "If you are not reasonably confident, you must set fields to null instead of guessing wildly."
            ),
        },
        {
            "role": "user",
            "content": (
                "From the following vehicle information, estimate the requested fields.\n"
                "Return a STRICT JSON object with exactly these keys:\n"
                "  msrp_usd_v0 (number or null),\n"
                "  engine_displacement_l (number in liters or null),\n"
                "  hp (number or null),\n"
                "  engine_configuration (string or null),\n"
                "  number_of_cylinders (integer or null),\n"
                "  supply_by_country (object mapping 2-letter country codes to approximate supply levels or null).\n\n"
                f"make: {make}\n"
                f"model: {model}\n"
                f"year: {year}\n"
                f"trim: {trim}\n"
            ),
        },
    ]

    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    raw_content = response.choices[0].message.content or "{}"

    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError:
        # Worst case, return empty structure
        data = {}

    # Normalize and ensure all keys exist
    specs: Dict[str, Any] = {
        "msrp_usd_v0": data.get("msrp_usd_v0"),
        "engine_displacement_l": data.get("engine_displacement_l"),
        "hp": data.get("hp"),
        "engine_configuration": data.get("engine_configuration"),
        "number_of_cylinders": data.get("number_of_cylinders"),
        "supply_by_country": data.get("supply_by_country"),
    }

    return specs


def update_gamme_row_from_llm(
    make: str,
    model: str,
    year: int | str,
    trim: str,
) -> Dict[str, Any]:
    """
    Main entrypoint:

    - Build gamme_id = MAKEMODELTRIMYEAR
    - Ask OpenAI for specs (MSRP, engine, supply)
    - Upsert into 'gammes' table with the new values

    Returns the dict that was upserted.
    """
    supabase = get_supabase_client()

    gamme_id = _build_gamme_id(make, model, year, trim)
    specs = _fetch_gamme_specs_from_llm(make, model, year, trim)

    # Map directly to your 'gammes' table column names.
    # Make sure these column names match your schema.
    payload = {
        "gamme_id": gamme_id,
        "make": make,
        "model": model,
        "year": int(year),
        "trim": trim,
        "msrp_usd_v0": specs.get("msrp_usd_v0"),
        "engine_displacement_l": specs.get("engine_displacement_l"),
        "hp": specs.get("hp"),
        "engine_configuration": specs.get("engine_configuration"),
        "number_of_cylinders": specs.get("number_of_cylinders"),
        # Assuming `supply_by_country` is a JSONB column in Supabase
        "supply_by_country": specs.get("supply_by_country"),
    }

    # Use upsert so it works whether the gamme already exists or not
    (
        supabase.table("gammes")
        .upsert(payload, on_conflict="gamme_id")
        .execute()
    )

    return payload

