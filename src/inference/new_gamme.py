# src/inference/new_gamme.py
from __future__ import annotations

import json
from typing import Dict, Any

from openai import OpenAI
from src.data.supabase_client import get_supabase_client

# LLM model
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
    Ask OpenAI for:
      - msrp_usd_v0        -> V0
      - engine_displacement_l -> engine_displacement
      - hp
      - engine_configuration
      - number_of_cylinders
      - supply_by_country (JSON dict, later aggregated to bigint for DB)
    """

    messages = [
        {
            "role": "system",
            "content": (
                "You are an automotive data assistant. "
                "Given make, model, year and trim, estimate specs and MSRP. "
                "If uncertain, answer null instead of hallucinating."
            ),
        },
        {
            "role": "user",
            "content": (
                "Return STRICT JSON with EXACT keys:\n"
                "  msrp_usd_v0,\n"
                "  engine_displacement_l,\n"
                "  hp,\n"
                "  engine_configuration,\n"
                "  number_of_cylinders,\n"
                "  supply_by_country (object: country_code -> supply number).\n\n"
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

    raw = response.choices[0].message.content or "{}"

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}

    return {
        "msrp_usd_v0": data.get("msrp_usd_v0"),
        "engine_displacement_l": data.get("engine_displacement_l"),
        "hp": data.get("hp"),
        "engine_configuration": data.get("engine_configuration"),
        "number_of_cylinders": data.get("number_of_cylinders"),
        "supply_by_country": data.get("supply_by_country"),
    }


def update_gamme_row_from_llm(
    make: str,
    model: str,
    year: int | str,
    trim: str,
) -> Dict[str, Any]:
    """
    Create or update the 'gammes' row with:
      - gamme_id = MAKEMODELTRIMYEAR (TEXT)
      - Columns: V0, engine_displacement, hp, engine_configuration,
                 number_of_cylinders, supply_by_country

    Assumptions:
      - gammes.gamme_id is TEXT (NOT NULL, PRIMARY KEY / UNIQUE)
      - gammes has columns: gamme_id, make, model, year, trim,
                            V0, engine_displacement, hp,
                            engine_configuration, number_of_cylinders,
                            supply_by_country (BIGINT)
    """

    supabase = get_supabase_client()
    year_int = int(year)

    gamme_id = _build_gamme_id(make, model, year_int, trim)

    # 1) Fetch specs from LLM
    specs = _fetch_gamme_specs_from_llm(make, model, year_int, trim)

    # 2) Adapt supply_by_country (dict) -> bigint for DB
    raw_supply = specs.get("supply_by_country")
    supply_agg = None

    if isinstance(raw_supply, dict):
        try:
            supply_agg = int(
                sum(v for v in raw_supply.values() if isinstance(v, (int, float)))
            )
        except Exception:
            supply_agg = None
    elif isinstance(raw_supply, (int, float, str)):
        try:
            supply_agg = int(raw_supply)
        except Exception:
            supply_agg = None

    # 3) Build payload, aligned with your DB columns
    payload: Dict[str, Any] = {
        "gamme_id": gamme_id,
        "make": make,
        "model": model,
        "year": year_int,
        "trim": trim,
        "V0": specs.get("msrp_usd_v0"),
        "engine_displacement": specs.get("engine_displacement_l"),
        "hp": specs.get("hp"),
        "engine_configuration": specs.get("engine_configuration"),
        "number_of_cylinders": specs.get("number_of_cylinders"),
        "supply_by_country": supply_agg,
    }

    # 4) Upsert on gamme_id (TEXT)
    result = (
        supabase.table("gammes")
        .upsert(payload, on_conflict="gamme_id")
        .execute()
    )

    # Attach raw specs for debugging in notebook
    payload["_raw_specs"] = specs
    payload["_supabase_result"] = result.data

    return payload
