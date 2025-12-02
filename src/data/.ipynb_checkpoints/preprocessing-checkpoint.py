from __future__ import annotations

from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
import datetime as dt


# ============================================================
#  Feature configuration
# ============================================================

# Features that describe the vehicle itself (for the Parameter NN)
VEHICLE_FEATURES: List[str] = [
    "make",
    "model",
    "year",
    "trim",
    "mileage",
    "fuel_type",
    "drivetrain",
    "color",
    "interior_color",
    "selected_options",   # likely list/JSON -> will need encoding later
    "aftermarket_mods",   # same
    "paddock_score",
]

# Features that describe the market context (for the residual/market model)
MARKET_FEATURES: List[str] = [
    "city",
    "state",
    # These will likely come from MarketCheck or your own aggregations
    "market_median_price",
    "market_min_price",
    "market_max_price",
    "market_supply",
    "market_days_on_market",
    "season",  # e.g. month or a categorical season label
]

# Target column for training today's value model
TARGET_COLUMN: str = "current_value"


# ============================================================
#  Basic cleaning helpers
# ============================================================

def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Safely convert a Pandas Series to numeric, coercing errors to NaN."""
    return pd.to_numeric(series, errors="coerce")


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add standard derived columns used by models, e.g. age_years.
    Does not modify the original df (returns a copy).
    """
    df = df.copy()

    # Age in years from 'year'
    if "year" in df.columns:
        current_year = dt.datetime.now().year
        df["year"] = _coerce_numeric(df["year"]).astype("Int64")
        df["age_years"] = current_year - df["year"]
    else:
        df["age_years"] = np.nan

    # Ensure mileage is numeric
    if "mileage" in df.columns:
        df["mileage"] = _coerce_numeric(df["mileage"])
    else:
        df["mileage"] = np.nan

    return df


# ============================================================
#  Feature selection
# ============================================================

def select_vehicle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with only the vehicle-intrinsic features + derived ones.
    Missing feature columns are silently skipped (but you can log if you want).
    """
    df = add_derived_columns(df)

    feature_cols = []
    for col in VEHICLE_FEATURES:
        if col in df.columns:
            feature_cols.append(col)
        # else: could log a warning here if you want

    # Always include derived 'age_years' if present
    if "age_years" in df.columns:
        feature_cols.append("age_years")

    return df[feature_cols].copy()


def select_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with only the market-context features.
    Missing columns are skipped.
    """
    feature_cols = [col for col in MARKET_FEATURES if col in df.columns]
    return df[feature_cols].copy()


# ============================================================
#  Training dataset builder
# ============================================================

def build_training_dataset(
    df: pd.DataFrame,
    require_target: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
    """
    Given a raw vehicles DataFrame (from Supabase, joined with MarketCheck, etc.),
    return (X_vehicle, X_market, y).

    - X_vehicle: features for the Parametric / NN Parameters model
    - X_market:  features for the Market Residual model
    - y:         target = current_value (or None if require_target=False and missing)

    Rows with missing TARGET_COLUMN are dropped if require_target=True.
    """

    df = df.copy()
    df = add_derived_columns(df)

    # Target
    if TARGET_COLUMN in df.columns:
        y = _coerce_numeric(df[TARGET_COLUMN])
    else:
        if require_target:
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' not found in DataFrame columns: {list(df.columns)}"
            )
        else:
            y = None

    # Drop rows with missing target if required
    if y is not None and require_target:
        mask = y.notna()
        df = df[mask]
        y = y[mask]

    # Vehicle features
    X_vehicle = select_vehicle_features(df)

    # Market features
    X_market = select_market_features(df)

    return X_vehicle, X_market, y
