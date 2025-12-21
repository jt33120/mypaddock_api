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


# -----------------------------
# Mileage targets / buckets
# -----------------------------

def _mile_targets_forward_heavy(m: float) -> List[float]:
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
    out: List[float] = []
    seen = set()
    for t in targets:
        tt = int(_clamp_miles(t))
        if tt not in seen:
            seen.add(tt)
            out.append(float(tt))
    return out


def _probe_targets_mvp(m: float) -> List[float]:
    m = _clamp_miles(m)
    targets = [
        0.5 * m,
        0.75 * m,
        m + 1_000,
        m + 5_000,
        m + 10_000,
        m + 20_000,
        m + 30_000,
        m + 50_000,
        m + 70_000,
        m + 100_000,
        m + 130_000,
    ]
    out: List[float] = []
    seen = set()
    for t in targets:
        tt = int(_clamp_miles(t))
        if tt not in seen:
            seen.add(tt)
            out.append(float(tt))
    return out


def _allocate_quota(total: int, weights: List[int]) -> List[int]:
    if total <= 0:
        return [0 for _ in weights]
    wsum = sum(max(0, w) for w in weights) or 1
    base = [int(total * (max(0, w) / wsum)) for w in weights]
    used = sum(base)
    rem = total - used
    i = len(base) - 1
    while rem > 0 and i >= 0:
        base[i] += 1
        rem -= 1
        i -= 1
        if i < 0:
            i = len(base) - 1
    return base


def _pick_comparables_by_targets(
    comparables: List[Dict[str, Any]],
    targets: List[float],
    per_target: List[int],
) -> List[Dict[str, Any]]:
    if not comparables:
        return []

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

    def pick_nearest(target: float, k: int) -> Tuple[List[Dict[str, Any]], int]:
        if k <= 0:
            return [], 0

        pool = [c for c in candidates if c.get("listing_id") not in selected_ids]
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
            c.pop("_miles_f", None)
            out_local.append(c)

        remaining = k - len(out_local)
        return out_local, remaining

    leftovers: List[int] = []
    for t, k in zip(targets, per_target):
        picked, rem = pick_nearest(t, k)
        selected.extend(picked)
        leftovers.append(rem)

    for idx in range(len(targets) - 1, -1, -1):
        rem = leftovers[idx]
        if rem <= 0:
            continue
        picked, _ = pick_nearest(targets[idx], rem)
        selected.extend(picked)

    return selected


# -----------------------------
# Grouping helpers
# -----------------------------

def _vehicle_group_key(row: Dict[str, Any]) -> str:
    """
    Stable grouping key for "same vehicle".
    Priority:
      1) vehicles.vin (best)
      2) vehicles.vehicle_id
      3) base_row.vehicle_id
    """
    vehicles = row.get("vehicles") or {}
    vin = vehicles.get("vin")
    if vin:
        return f"vin:{vin}"
    vvid = vehicles.get("vehicle_id")
    if vvid:
        return f"veh:{vvid}"
    return f"row:{row.get('vehicle_id')}"


def _parse_date_iso(d: Any) -> Optional[datetime]:
    if not d:
        return None
    try:
        # expecting YYYY-MM-DD
        return datetime.strptime(str(d), "%Y-%m-%d")
    except Exception:
        return None


def _pick_latest_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pick most recent row by 'date' (YYYY-MM-DD). If parsing fails, fall back
    to last element.
    """
    if not rows:
        raise ValueError("rows empty")
    best = rows[-1]
    best_dt = _parse_date_iso(best.get("date"))
    for r in rows:
        dt = _parse_date_iso(r.get("date"))
        if best_dt is None:
            # if best unknown, accept any known dt
            if dt is not None:
                best = r
                best_dt = dt
        else:
            if dt is not None and dt > best_dt:
                best = r
                best_dt = dt
    return best


# -----------------------------
# MarketCheck I/O
# -----------------------------

def _fetch_mc_comparables_for_vehicle(
    base_row: Dict[str, Any],
    max_similars: int = 200,
) -> tuple[Optional[float], List[Dict[str, Any]]]:
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
        mc_price = data.get("marketcheck_price")
        listings = (data.get("comparables") or {}).get("listings") or []

        simplified: List[Dict[str, Any]] = []
        for lst in listings:
            build = lst.get("build", {})
            sim_make = build.get("make") or lst.get("make") or make
            sim_model = build.get("model") or lst.get("model") or model
            sim_year = build.get("year") or lst.get("year") or year
            sim_trim = build.get("trim") or lst.get("trim") or trim
            sim_miles = lst.get("miles") or lst.get("mileage") or mileage
            sim_id = lst.get("id") or lst.get("listing_id")

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
    source: str,
) -> Dict[str, Any]:
    vehicles = base_row.get("vehicles") or {}
    gammes = (vehicles or {}).get("gammes") or {}
    today = datetime.utcnow().date().isoformat()

    prefix = source  # e.g. "mc_comp" or "mc_probe"
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


# -----------------------------
# Main fetch
# -----------------------------

def fetch_training_rows_since(
    since_date: str,
    use_marketcheck: bool = True,
    max_mc_vehicles: Optional[int] = None,

    max_similars_per_vehicle: int = 200,
    max_calls_per_vehicle: int = 80,

    enable_subject_probes: bool = True,
    desired_probe_points: int = 11,

    use_listing_price_if_present: bool = False,
) -> List[Dict[str, Any]]:
    """
    NEW behavior:
      - Always returns ALL base_rows since since_date (real points).
      - MarketCheck augmentation runs ONLY on the most recent base_row per vehicle (grouped by VIN/vehicle_id).
        This prevents multiplying probes/comps by daily fetch frequency.
    """

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

    if not use_marketcheck or not base_rows:
        return base_rows

    # ---- Pick ONLY the most recent row per vehicle group for MarketCheck augmentation ----
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in base_rows:
        k = _vehicle_group_key(r)
        grouped.setdefault(k, []).append(r)

    # One "MC anchor row" per vehicle group
    mc_anchor_rows: List[Dict[str, Any]] = []
    for k, rows in grouped.items():
        latest = _pick_latest_row(rows)
        mc_anchor_rows.append(latest)

    # Optional throttle (applies to number of vehicles, not number of base_rows)
    if max_mc_vehicles is not None:
        mc_anchor_rows = mc_anchor_rows[:max_mc_vehicles]

    mc_rows: List[Dict[str, Any]] = []

    # ---- Run MC augmentation only for anchor rows ----
    for idx, base_row in enumerate(mc_anchor_rows):
        vehicles = base_row.get("vehicles") or {}
        gammes = (vehicles or {}).get("gammes") or {}
        base_mileage = base_row.get("mileage")
        vin = vehicles.get("vin")

        print(
            f"[MC] Processing anchor idx={idx}, group={_vehicle_group_key(base_row)}, "
            f"vehicle_id={base_row.get('vehicle_id')}, vin={vin}, date={base_row.get('date')}"
        )

        if base_mileage is None:
            print(f"[MC] Skip anchor: no mileage for vehicle_id={base_row.get('vehicle_id')}")
            continue

        calls_used = 0

        # 1) One comparables call (pool)
        _mc_price_subject, comparables_pool = _fetch_mc_comparables_for_vehicle(
            base_row,
            max_similars=max_similars_per_vehicle,
        )
        calls_used += 1

        if not comparables_pool and not enable_subject_probes:
            print(f"[MC] No comparables and probes disabled for anchor vehicle_id={base_row.get('vehicle_id')}")
            continue

        # 2) Decide probe targets and comp quota under cap
        probe_targets: List[float] = []
        if enable_subject_probes and vin:
            probe_targets = _probe_targets_mvp(float(base_mileage))[: max(0, desired_probe_points)]

        probe_budget = min(len(probe_targets), max(0, max_calls_per_vehicle - calls_used))
        comp_budget = max(0, max_calls_per_vehicle - calls_used - probe_budget)

        # 3) Select comps by buckets with forward-heavy weighting
        selected_comps: List[Dict[str, Any]] = []
        if comparables_pool and comp_budget > 0:
            targets = _mile_targets_forward_heavy(float(base_mileage))
            weights = [1, 1, 2, 2, 3, 3, 3]
            per_target = _allocate_quota(comp_budget, weights)
            selected_comps = _pick_comparables_by_targets(
                comparables=comparables_pool,
                targets=targets,
                per_target=per_target,
            )

        # 4) Price selected comps
        if selected_comps and comp_budget > 0:
            city = vehicles.get("city")
            state = vehicles.get("state")
            zip_code = vehicles.get("zip")
            dealer_type = vehicles.get("dealer_type") or "independent"

            priced_count = 0
            for comp in selected_comps:
                if priced_count >= comp_budget or calls_used >= max_calls_per_vehicle:
                    break

                raw = comp.get("raw") or {}

                if use_listing_price_if_present:
                    observed = raw.get("price") or raw.get("list_price") or raw.get("internet_price")
                    if observed is not None:
                        try:
                            price_val = float(observed)
                            spec_obs = {
                                "listing_id": comp.get("listing_id"),
                                "vin": raw.get("vin"),
                                "make": comp["make"],
                                "model": comp["model"],
                                "year": comp["year"],
                                "trim": comp.get("trim"),
                                "mileage": comp.get("mileage", 0.0),
                            }
                            mc_rows.append(_build_mc_training_row(base_row, spec_obs, price_val, source="mc_comp"))
                            priced_count += 1
                            continue
                        except Exception:
                            pass

                spec = {
                    "listing_id": comp.get("listing_id"),
                    "vin": raw.get("vin"),
                    "make": comp["make"],
                    "model": comp["model"],
                    "year": comp["year"],
                    "trim": comp.get("trim"),
                    "mileage": comp.get("mileage", 0.0),
                    "city": city,
                    "state": state,
                    "zip": zip_code,
                    "dealer_type": dealer_type,
                }

                comp_price = _fetch_mc_price_for_spec(spec)
                calls_used += 1
                priced_count += 1

                if comp_price is None:
                    continue

                mc_rows.append(_build_mc_training_row(base_row, spec, comp_price, source="mc_comp"))

        # 5) Probes
        if probe_budget > 0 and vin and calls_used < max_calls_per_vehicle:
            city = vehicles.get("city")
            state = vehicles.get("state")
            zip_code = vehicles.get("zip")
            dealer_type = vehicles.get("dealer_type") or "independent"

            for t in probe_targets[:probe_budget]:
                if calls_used >= max_calls_per_vehicle:
                    break

                probe_spec = {
                    "listing_id": None,
                    "vin": vin,
                    "make": gammes.get("make"),
                    "model": gammes.get("model"),
                    "year": gammes.get("year"),
                    "trim": gammes.get("trim"),
                    "mileage": float(t),
                    "zip": zip_code,
                    "city": city,
                    "state": state,
                    "dealer_type": dealer_type,
                }

                probe_price = _fetch_mc_price_for_spec(probe_spec)
                calls_used += 1

                if probe_price is None:
                    continue

                mc_rows.append(_build_mc_training_row(base_row, probe_spec, probe_price, source="mc_probe"))

        print(
            f"[MC] Done anchor group={_vehicle_group_key(base_row)} calls_used={calls_used}/{max_calls_per_vehicle} "
            f"comps_selected={len(selected_comps)} probes_targeted={probe_budget}"
        )

    return base_rows + mc_rows
