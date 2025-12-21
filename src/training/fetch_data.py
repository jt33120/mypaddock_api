# src/training/fetch_data.py

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import os
import requests

from src.data.supabase_client import get_supabase_client

# Base URLs for MarketCheck "comparables" + "MarketCheck Price™"
MC_BASE_PREDICT = "https://api.marketcheck.com/v2/predict/car/us/marketcheck_price"
MC_BASE_COMPARABLES = f"{MC_BASE_PREDICT}/comparables"


def _get_marketcheck_api_key() -> Optional[str]:
    return os.getenv("MARKETCHECK_API_KEY") or os.getenv("VITE_MARKETCHECK_API_KEY")


def _clamp_miles(x: float, lo: float = 0.0, hi: float = 300_000.0) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _mile_targets_forward_heavy(m: float) -> List[float]:
    """
    Your requested forward-heavy mileage targets:
      a few past points + many future points around and beyond m.
    """
    m = _clamp_miles(m)
    targets = [
        0.5 * m,
        0.75 * m,
        m + 1_000,
        m + 5_000,
        m + 10_000,
        m + 50_000,
        m + 100_000,
    ]
    # Deduplicate and clamp
    out: List[float] = []
    seen = set()
    for t in targets:
        tt = int(_clamp_miles(t))
        if tt not in seen:
            seen.add(tt)
            out.append(float(tt))
    return out


def _pick_comparables_by_targets(
    comparables: List[Dict[str, Any]],
    targets: List[float],
    per_target: List[int],
    subject_mileage: float,
) -> List[Dict[str, Any]]:
    """
    Select comparables in a forward-heavy way:
      - For each mileage target, pick up to N nearest unique listings.

    If some targets can't be filled (not enough comps), we redistribute leftover
    quota into later (more future) targets.

    This works even if MarketCheck doesn't return a perfect mileage spread.
    """
    if not comparables:
        return []

    # Prepare candidates (only those with a valid listing_id + mileage)
    candidates: List[Dict[str, Any]] = []
    for c in comparables:
        lid = c.get("listing_id")
        miles = c.get("mileage")
        if not lid:
            continue
        try:
            miles_f = float(miles)
        except Exception:
            continue
        c2 = dict(c)
        c2["_miles_f"] = miles_f
        candidates.append(c2)

    if not candidates:
        return []

    selected: List[Dict[str, Any]] = []
    selected_ids = set()

    # Helper: pick nearest candidates to a target, excluding already selected
    def pick_nearest(target: float, k: int) -> Tuple[List[Dict[str, Any]], int]:
        if k <= 0:
            return [], 0

        # sort by absolute distance to target
        pool = [
            c for c in candidates
            if c.get("listing_id") not in selected_ids
        ]
        if not pool:
            return [], k

        pool.sort(key=lambda c: abs(c["_miles_f"] - target))

        out_local: List[Dict[str, Any]] = []
        for c in pool:
            if len(out_local) >= k:
                break
            lid = c.get("listing_id")
            if lid in selected_ids:
                continue
            selected_ids.add(lid)
            # clean internal helper key
            c.pop("_miles_f", None)
            out_local.append(c)

        remaining = k - len(out_local)
        return out_local, remaining

    # First pass: try to fill each target's quota
    leftovers: List[int] = []
    for t, k in zip(targets, per_target):
        picked, rem = pick_nearest(t, k)
        selected.extend(picked)
        leftovers.append(rem)

    # Redistribute any leftovers into later targets (more future)
    # (priority: last targets first are more "future"; but we add in forward order
    # by simply looping from the end and picking more near that target)
    for idx in range(len(targets) - 1, -1, -1):
        rem = leftovers[idx]
        if rem <= 0:
            continue
        picked, _rem2 = pick_nearest(targets[idx], rem)
        selected.extend(picked)

    # Optional: ensure we didn't accidentally include "too close to subject" only.
    # (Keeping as-is: you're explicitly asking for forward-heavy targets.)
    return selected


def _fetch_mc_comparables_for_vehicle(
    base_row: Dict[str, Any],
    max_similars: int = 200,  # request a larger pool so we can stratify
) -> tuple[Optional[float], List[Dict[str, Any]]]:
    """
    Fetch a pool of MarketCheck comparables for a vehicle.
    NOTE: We keep returning mc_price for compatibility, but we no longer use it
    to create a subject synthetic row (to avoid doubling).
    """

    api_key = _get_marketcheck_api_key()
    if not api_key:
        print("[MC] No MARKETCHECK_API_KEY configured; skipping ALL comparables.")
        return None, []

    vehicles = base_row.get("vehicles") or {}
    gammes = (vehicles or {}).get("gammes") or {}

    vehicle_id = base_row.get("vehicle_id")
    vin = vehicles.get("vin")
    mileage = base_row.get("mileage")
    dealer_type = vehicles.get("dealer_type") or "independent"

    city = vehicles.get("city")
    state = vehicles.get("state")
    zip_code = vehicles.get("zip")

    make = gammes.get("make")
    model = gammes.get("model")
    year = gammes.get("year")
    trim = gammes.get("trim")

    # --- Required checks ---
    if not vin:
        print(f"[MC] Skipping vehicle (no VIN). vehicle_id={vehicle_id}, vin={vin}")
        return None, []

    if mileage is None:
        print(f"[MC] Skipping vehicle (no mileage). vehicle_id={vehicle_id}, vin={vin}")
        return None, []

    if not (zip_code or (city and state)):
        print(
            f"[MC] Skipping vehicle (no valid location). vehicle_id={vehicle_id}, "
            f"vin={vin}, city={city}, state={state}, zip={zip_code}"
        )
        return None, []

    if not (make and model and year):
        print(
            f"[MC] Skipping vehicle (missing make/model/year). vehicle_id={vehicle_id}, "
            f"vin={vin}, make={make}, model={model}, year={year}"
        )
        return None, []

    params = {
        "api_key": api_key,
        "vin": vin,
        "miles": mileage,
        "dealer_type": dealer_type,
        "rows": max_similars,
    }

    if zip_code:
        params["zip"] = zip_code
    else:
        params["city"] = city
        params["state"] = state

    print(
        f"[MC] Calling MarketCheck Price (comparables) for vehicle_id={vehicle_id}, vin={vin}. "
        f"Params={params}"
    )

    try:
        resp = requests.get(MC_BASE_COMPARABLES, params=params, timeout=10)
        if resp.status_code != 200:
            print(
                f"[MC] comparables error {resp.status_code} for vehicle_id={vehicle_id}, "
                f"vin={vin}: {resp.text[:200]}"
            )
            return None, []

        data = resp.json()

        mc_price = data.get("marketcheck_price")  # kept for compatibility/logging
        comps_block = data.get("comparables") or {}
        listings = comps_block.get("listings") or []

        simplified: List[Dict[str, Any]] = []
        for lst in listings:
            build = lst.get("build", {})
            sim_make = build.get("make") or lst.get("make") or make
            sim_model = build.get("model") or lst.get("model") or model
            sim_year = build.get("year") or lst.get("year") or year
            sim_trim = build.get("trim") or lst.get("trim") or trim
            sim_miles = lst.get("miles") or lst.get("mileage") or mileage
            sim_id = lst.get("id") or lst.get("listing_id")

            # Skip if year can't be int
            try:
                sim_year_i = int(sim_year)
            except Exception:
                continue

            simplified.append(
                {
                    "listing_id": sim_id,
                    "make": sim_make,
                    "model": sim_model,
                    "year": sim_year_i,
                    "trim": sim_trim,
                    "mileage": float(sim_miles) if sim_miles is not None else 0.0,
                    "raw": lst,
                }
            )

        return mc_price, simplified

    except Exception as e:
        print(f"[MC] comparables exception for vehicle_id={vehicle_id}, vin={vin}: {e}")
        return None, []


def _fetch_mc_price_for_spec(spec: Dict[str, Any]) -> Optional[float]:
    api_key = _get_marketcheck_api_key()
    if not api_key:
        return None

    listing_id = spec.get("listing_id")
    vin = spec.get("vin")
    miles = spec.get("mileage")
    dealer_type = spec.get("dealer_type") or "independent"
    zip_code = spec.get("zip")
    city = spec.get("city")
    state = spec.get("state")

    missing = []
    if not vin:
        missing.append("vin")
    if miles is None:
        missing.append("mileage")
    if not (zip_code or (city and state)):
        missing.append("location (zip or city+state)")

    if missing:
        print(
            f"[MC] price skip for listing_id={listing_id}: missing {', '.join(missing)}. "
            f"spec={spec}"
        )
        return None

    params = {
        "api_key": api_key,
        "vin": vin,
        "miles": miles,
        "dealer_type": dealer_type,
    }

    if zip_code:
        params["zip"] = zip_code
    else:
        params["city"] = city
        params["state"] = state

    # Optional: enrich with make/model/year/trim
    if spec.get("make"):
        params["make"] = spec["make"]
    if spec.get("model"):
        params["model"] = spec["model"]
    if spec.get("year"):
        params["year"] = spec["year"]
    if spec.get("trim"):
        params["trim"] = spec["trim"]

    try:
        resp = requests.get(MC_BASE_PREDICT, params=params, timeout=10)
        if resp.status_code != 200:
            print(
                f"[MC] price error {resp.status_code} for listing_id={listing_id}, "
                f"vin={vin}: {resp.text[:200]}"
            )
            return None

        data = resp.json()
        price = data.get("marketcheck_price") or data.get("predicted_price") or data.get("price")

        if price is None:
            print(
                f"[MC] price response missing price field for listing_id={listing_id}, "
                f"vin={vin}, data_keys={list(data.keys())}"
            )
            return None

        return float(price)

    except Exception as e:
        print(f"[MC] price exception for listing_id={listing_id}, vin={vin}: {e}")
        return None


def _build_mc_training_row(
    base_row: Dict[str, Any],
    spec: Dict[str, Any],
    mc_price: float,
    source: str = "marketcheck",
) -> Dict[str, Any]:
    """
    Create a synthetic training row (same "shape" as Supabase query result)
    from a MarketCheck comparable or probe + its MarketCheck Price™.
    """
    vehicles = base_row.get("vehicles") or {}
    gammes = (vehicles or {}).get("gammes") or {}

    today = datetime.utcnow().date().isoformat()

    # For traceability, use different prefixes
    prefix = "mc"
    if source == "mc_probe":
        prefix = "mc_probe"
    elif source == "mc_comp":
        prefix = "mc_comp"

    return {
        "vehicle_id": f"{prefix}_{spec.get('listing_id') or spec.get('vin') or spec.get('make')}_{spec.get('year')}",
        "date": today,
        "value": mc_price,
        "mileage": spec.get("mileage", 0.0),
        "source": source,
        "vehicles": {
            "vehicle_id": None,
            "gamme_id": vehicles.get("gamme_id"),
            "type": vehicles.get("type"),
            "color": vehicles.get("color"),
            "transmission": vehicles.get("transmission"),
            "fuel_type": vehicles.get("fuel_type"),
            "drivetrain": vehicles.get("drivetrain"),
            "bodystyle": vehicles.get("bodystyle"),
            "interior_color": vehicles.get("interior_color"),
            "paddock_score": vehicles.get("paddock_score"),
            "city": vehicles.get("city"),
            "state": vehicles.get("state"),
            "doors_numbers": vehicles.get("doors_numbers"),
            "gammes": {
                "gamme_id": gammes.get("gamme_id"),
                "make": spec["make"],
                "model": spec["model"],
                "year": spec["year"],
                "trim": spec.get("trim"),
                "V0": gammes.get("V0"),
                "hp": gammes.get("hp"),
                "engine_displacement": gammes.get("engine_displacement"),
                "engine_configuration": gammes.get("engine_configuration"),
                "number_of_cylinders": gammes.get("number_of_cylinders"),
                "supply_by_country": gammes.get("supply_by_country"),
                "delta": gammes.get("delta"),
                "alpha": gammes.get("alpha"),
                "k_a": gammes.get("k_a"),
                "k_m": gammes.get("k_m"),
                "L": gammes.get("L"),
                "c": gammes.get("c"),
            },
        },
    }


def fetch_training_rows_since(
    since_date: str,
    use_marketcheck: bool = True,
    # remove the limit by default; keep optional throttle if you want to pass it
    max_mc_vehicles: Optional[int] = None,
    # how many comps to REQUEST from MarketCheck (pool size)
    max_similars_per_vehicle: int = 200,
    # quotas for your forward-heavy milestones
    # targets: [0.5m, 0.75m, m+1k, m+5k, m+10k, m+50k, m+100k]
    per_target_quota: Optional[List[int]] = None,
    # optional: create "same VIN" future probes at those milestone mileages
    enable_subject_probes: bool = True,
    # how many probes per target mileage to add (usually 1 is enough)
    probes_per_target: int = 1,
) -> List[Dict[str, Any]]:

    client = get_supabase_client()

    resp = (
        client.table("vehicle_timeseries")
        .select(
            "vehicle_id, date, value, mileage,"
            "vehicles!inner("
            "  vehicle_id, gamme_id, vin, type, color, transmission, fuel_type, drivetrain,"
            "  bodystyle, interior_color, paddock_score,"
            "  city, state, zip, dealer_type, doors_numbers,"
            "  gammes!inner("
            "    gamme_id, make, model, year, trim, V0, hp, engine_displacement,"
            "    engine_configuration, number_of_cylinders, supply_by_country,"
            "    delta, alpha, k_a, k_m, L, c"
            "  )"
            ")"
        )
        .gte("date", since_date)
        .execute()
    )

    base_rows = resp.data or []
    for r in base_rows:
        r["source"] = "supabase"

    if not use_marketcheck:
        return base_rows

    # Default quotas: future-heavy
    # You suggested something like: few past, some around, more future.
    # Here we map to the 7 targets:
    #   0.5m, 0.75m, +1k, +5k, +10k, +50k, +100k
    # Feel free to tune.
    if per_target_quota is None:
        per_target_quota = [2, 2, 5, 5, 10, 10, 10]  # total 44 comps if available

    mc_rows: List[Dict[str, Any]] = []

    for idx, base_row in enumerate(base_rows):
        if max_mc_vehicles is not None and idx >= max_mc_vehicles:
            break

        vehicles = base_row.get("vehicles") or {}
        gammes = (vehicles or {}).get("gammes") or {}

        vin = vehicles.get("vin")
        base_mileage = base_row.get("mileage")

        print(f"[MC] Processing idx={idx}, vehicle_id={base_row.get('vehicle_id')}, vin={vin}")

        mc_price_subject, comparables_pool = _fetch_mc_comparables_for_vehicle(
            base_row,
            max_similars=max_similars_per_vehicle,
        )

        if not comparables_pool and not enable_subject_probes:
            print(f"[MC] No MarketCheck comparables and probes disabled for vehicle_id={base_row.get('vehicle_id')}, vin={vin}")
            continue

        # ---- A) Mileage-stratified comparable selection (forward-heavy) ----
        selected_comps: List[Dict[str, Any]] = []
        if comparables_pool and base_mileage is not None:
            targets = _mile_targets_forward_heavy(float(base_mileage))
            # Ensure quota list matches target list length
            qt = per_target_quota[: len(targets)]
            if len(qt) < len(targets):
                qt = qt + [qt[-1]] * (len(targets) - len(qt))

            selected_comps = _pick_comparables_by_targets(
                comparables=comparables_pool,
                targets=targets,
                per_target=qt,
                subject_mileage=float(base_mileage),
            )

        # Price each selected comparable and add synthetic rows
        if selected_comps:
            city = vehicles.get("city")
            state = vehicles.get("state")

            for comp in selected_comps:
                raw = comp.get("raw") or {}

                spec = {
                    "listing_id": comp.get("listing_id"),
                    "vin": raw.get("vin"),  # depends on MC listing including VIN
                    "make": comp["make"],
                    "model": comp["model"],
                    "year": comp["year"],
                    "trim": comp.get("trim"),
                    "mileage": comp.get("mileage", 0.0),
                    # reuse subject context for location + dealer_type
                    "city": city,
                    "state": state,
                    "zip": vehicles.get("zip"),
                    "dealer_type": vehicles.get("dealer_type") or "independent",
                }

                comp_price = _fetch_mc_price_for_spec(spec)
                if comp_price is None:
                    continue

                mc_row = _build_mc_training_row(base_row, spec, comp_price, source="mc_comp")
                mc_rows.append(mc_row)

        # ---- B) Subject VIN forward probes (optional) ----
        # NOTE: This DOES NOT create a "subject synthetic row at current mileage".
        # It only creates *future/past milestone probe points*.
        if enable_subject_probes and base_mileage is not None:
            subj_vin = vehicles.get("vin")
            if not subj_vin:
                # can't probe without VIN
                continue

            city = vehicles.get("city")
            state = vehicles.get("state")
            zip_code = vehicles.get("zip")

            # We'll probe at the same target mileages (including the two past points)
            targets = _mile_targets_forward_heavy(float(base_mileage))

            for t in targets:
                # optionally skip "past" targets if you only want future:
                # if t < float(base_mileage): continue

                for _ in range(max(1, probes_per_target)):
                    probe_spec = {
                        "listing_id": None,
                        "vin": subj_vin,
                        "make": gammes.get("make"),
                        "model": gammes.get("model"),
                        "year": gammes.get("year"),
                        "trim": gammes.get("trim"),
                        "mileage": float(t),
                        "zip": zip_code,
                        "city": city,
                        "state": state,
                        "dealer_type": vehicles.get("dealer_type") or "independent",
                    }

                    probe_price = _fetch_mc_price_for_spec(probe_spec)
                    if probe_price is None:
                        continue

                    probe_row = _build_mc_training_row(base_row, probe_spec, probe_price, source="mc_probe")
                    mc_rows.append(probe_row)

    return base_rows + mc_rows
