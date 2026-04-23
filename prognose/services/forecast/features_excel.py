"""Feature engineering for the Excel-based GBDT training pipeline.

This module sits between raw training data (output of prepare_training_data.py)
and the model training step (train_gbdt.py). Its responsibilities are:

  1. Define exactly which 40 features are used by the LightGBM model (FEATURE_COLUMNS).
  2. Build a supervised single-output dataset from the prepared Parquet file, including
     the forecast target variable y = occupancy at t+horizon.
  3. Generate chronologically correct walk-forward cross-validation fold splits.

Design decisions:
  - Feature set is versioned (FEATURE_SET_VERSION = "excel_v1") so that a model bundle
    always knows which features it was trained on.
  - Target is clipped to [0, capacity] during training so the model never learns to
    predict physically impossible values.
  - Walk-forward splits are timestamp-based (not index-based) to be robust against data
    gaps or filtered rows.

Key symbols:
    FEATURE_COLUMNS: Ordered list of the 40 feature names fed to LightGBM.
    SupervisedDataset: Dataclass holding X, y, timestamps, and metadata.
    build_supervised_dataset: Create the target variable and align features/target.
    walk_forward_splits: Generate n-fold expanding-window CV splits.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Version string embedded in every saved model bundle so that training code
# and inference code can verify they are using the same feature set.
FEATURE_SET_VERSION = "excel_v1"

# Physical upper bound on the number of occupants.  This value is used to
# clip the forecast target during training and must match the capacity constant
# in prepare_training_data.py and in the inference path of main.py.
CAPACITY_TOTAL = 84

# Features that must NOT be scaled by a StandardScaler or MinMaxScaler.
# Cyclic features already live in [-1, 1]; binary features are exactly {0, 1}.
# Scaling these would distort their meaning (e.g. sin(0) -> 0, sin(pi) -> -1
# are meaningful; scaling them would move the origin and lose periodicity).
PASSTHROUGH_FEATURES = [
    "minute_sin", "minute_cos", "dow_sin", "dow_cos",
    "day_of_year_sin", "day_of_year_cos",
    "is_weekday", "bridge_day", "winter_break",
    "weather_rainy", "weather_sunny", "is_partial_closure",
]

# ---------------------------------------------------------------------------
# FEATURE_COLUMNS — the canonical ordered list of 40 model inputs.
#
# Order matters: the model bundle stores this list in metadata_gbdt.json and
# the inference path in main.py must reconstruct features in this exact order.
# Any change here requires a full model retrain.
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    # ---- Excel-native factor scores (7 features) ----
    # Pre-computed by the study team; each encodes domain knowledge about
    # expected attendance at a given time under given conditions.
    "f_month",           # Month-level seasonal factor (e.g. exam period vs. holidays)
    "f_weekday",         # Weekday-level factor (Mon-Sat variations)
    "f_tod",             # Time-of-day factor (opening hours pattern)
    "f_weather",         # Weather impact score
    "f_bridge",          # Bridge-day factor (reduced attendance around public holidays)
    "efficiency",        # General efficiency/productivity score from the dataset
    "capacity_effective", # Actual usable capacity (may differ from nominal if areas closed)

    # ---- Binary context flags (5 features) ----
    # One-hot encoded indicators for special conditions.
    "bridge_day",          # 1 if this is a bridge day between a weekend and a holiday
    "winter_break",        # 1 if semester break (Christmas / semester gap)
    "weather_rainy",       # 1 if rainy weather (drives students indoors → higher occupancy)
    "weather_sunny",       # 1 if sunny weather (drives students outdoors → lower occupancy)
    "is_partial_closure",  # 1 if part of the library is closed (capacity reduced)

    # ---- Utilization rate (1 feature) ----
    "utilization_pct",  # occupancy / effective_capacity * 100; normalised load signal

    # ---- Temporal / cyclic encodings (8 features) ----
    # Sine/cosine pairs make time periodic so that, e.g., 23:59 and 00:01 are
    # numerically close in feature space (as they should be semantically).
    "minute_sin", "minute_cos",          # Time of day (period = 24h)
    "dow_sin", "dow_cos",                # Day of week (period = 7 days)
    "day_of_year_sin", "day_of_year_cos", # Season (period = 365 days)
    "hour_of_day",   # Raw hour (0-23); useful as a direct split threshold for trees
    "is_weekday",    # 1 = Mon-Fri, 0 = Sat/Sun; strong signal for student patterns

    # ---- Occupancy lag features (8 features) ----
    # Autoregressive inputs: past values of the target variable.  These are the
    # single most predictive feature group because occupancy is strongly
    # autocorrelated (the library is usually full/empty for stretches of time).
    "occupancy_lag_1",    # 15 min ago
    "occupancy_lag_2",    # 30 min ago
    "occupancy_lag_3",    # 45 min ago
    "occupancy_lag_4",    # 60 min ago (same offset as forecast horizon)
    "occupancy_lag_8",    # 2 h ago
    "occupancy_lag_16",   # 4 h ago
    "occupancy_lag_day",  # ~12 operating hours ago (48 × 15 min = same slot yesterday)
    "occupancy_lag_week", # ~72 h ago (288 × 15 min = same slot last working week)

    # ---- Rolling statistics (5 features) ----
    # Smooth out measurement noise and capture short-term momentum / volatility.
    "occupancy_roll_mean_4",   # Rolling mean over last 1 h (4 × 15 min)
    "occupancy_roll_mean_8",   # Rolling mean over last 2 h
    "occupancy_roll_mean_16",  # Rolling mean over last 4 h
    "occupancy_roll_std_4",    # Rolling std over last 1 h (volatility proxy)
    "occupancy_roll_std_8",    # Rolling std over last 2 h

    # ---- First-order differences (2 features) ----
    # Rate-of-change signals: positive = occupancy rising, negative = falling.
    "occupancy_diff_1",  # Change vs. 15 min ago
    "occupancy_diff_4",  # Change vs. 60 min ago (1-hour trend)

    # ---- Lecture proxy features (4 features) ----
    # Derived from the weekly DHBW lecture activity profile (weekday × hour lookup).
    # Captures that students tend to arrive after lectures end and leave before they start.
    "lecture_density_proxy",  # Mean active lectures at this (weekday, hour)
    "lecture_starts_proxy",   # Mean lecture starts in the next 60 min
    "lecture_ends_proxy",     # Mean lecture ends in the next 60 min
    "lecture_heavy_proxy",    # Mean count of 'heavy' parallel lecture slots (≥3)
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SupervisedDataset:
    """A fully prepared supervised dataset for single-step-ahead forecasting.

    Bundles the feature matrix X, target array y, and metadata so that training
    code and cross-validation code can pass everything in one object rather than
    separate arguments.

    Attributes:
        X: Feature matrix (n_samples × n_features), columns = feature_columns.
        y: Target array (n_samples,), dtype float32, clipped to [0, capacity].
        feature_columns: List of column names in X (same order as FEATURE_COLUMNS).
        target_name: String like "y_tplus_4" encoding the forecast horizon.
        timestamps: UTC timestamps for each row, used for time-based CV splitting.
    """
    X: pd.DataFrame
    y: np.ndarray
    feature_columns: list[str]
    target_name: str
    timestamps: pd.Series


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def load_training_data(parquet_path: str) -> pd.DataFrame:
    """Load the prepared Parquet training file and ensure UTC timestamps.

    Args:
        parquet_path: Path to the training_data.parquet file produced by
            prepare_training_data.py.

    Returns:
        Chronologically sorted DataFrame ready for build_supervised_dataset.
    """
    df = pd.read_parquet(parquet_path)
    # Ensure timestamps are timezone-aware UTC for consistent comparisons.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def build_supervised_dataset(
    df: pd.DataFrame,
    horizon_steps: int = 4,
    capacity: int = CAPACITY_TOTAL,
) -> SupervisedDataset:
    """Convert a feature DataFrame into a supervised learning dataset.

    The key transformation here is creating the forecast target: we shift the
    occupancy column forward by `horizon_steps` positions so that each row's
    label is the future occupancy the model should predict.

    Example for horizon_steps=4 (the default, H60):
      Row 0 (08:00): features from 08:00, target = occupancy at 09:00
      Row 1 (08:15): features from 08:15, target = occupancy at 09:15
      ...

    The target is clipped to [0, capacity] for two reasons:
      1. Physical impossibility: a room cannot hold more than its capacity.
      2. Consistency with inference: the inference path clips predictions to
         the same range, so training on clipped targets avoids a mismatch.

    Rows at the end of the series where the shifted target falls outside the
    DataFrame (NaN) are dropped, as are rows where any feature is NaN.

    Args:
        df: Training DataFrame from load_training_data or prepare_training_data.
        horizon_steps: Number of 15-min steps ahead to predict.
            4 = 60 min (H60, the primary production horizon).
        capacity: Physical room capacity used for target clipping.

    Returns:
        SupervisedDataset ready for train_gbdt.py.

    Raises:
        ValueError: If the DataFrame is empty, horizon_steps < 1, or no valid
            rows remain after NaN filtering.
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be >= 1")

    # Only include features that are actually present in the DataFrame.
    # This makes the function robust to partial feature sets during testing.
    available = [col for col in FEATURE_COLUMNS if col in df.columns]
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        print(f"[warn] missing features (will be skipped): {missing}")

    # Create the forecast target by shifting occupancy forward by horizon_steps.
    # shift(-4) means: "what will occupancy be 4 rows (= 60 min) into the future?"
    target_name = f"y_tplus_{horizon_steps}"
    target = df["occupancy"].shift(-horizon_steps)

    # Clip target to [0, capacity].  Clipping at training time ensures the model
    # never receives labels > capacity as ground truth, which would bias it towards
    # predicting over-capacity values.
    target = target.clip(lower=0, upper=capacity)

    # Build a boolean mask of rows where both features and target are non-NaN.
    # NaN in target occurs at the last horizon_steps rows (no future to shift into).
    # NaN in features occurs at the first rows (not enough history for lags).
    valid_mask = target.notna()
    for col in available:
        valid_mask &= df[col].notna()

    X = df.loc[valid_mask, available].copy()
    X.index = range(len(X))  # Reset to contiguous integers, preserving order.
    y = target[valid_mask].values.astype(np.float32)
    timestamps = df.loc[valid_mask, "timestamp"].reset_index(drop=True)

    if len(X) == 0:
        raise ValueError("No valid rows after target alignment and NaN removal")

    # Verify strictly non-decreasing timestamps — a necessary precondition for
    # walk-forward cross-validation.  Any reordering would cause data leakage.
    ts_vals = timestamps.values
    if not all(ts_vals[i] <= ts_vals[i + 1] for i in range(len(ts_vals) - 1)):
        raise ValueError("Timestamps are not chronologically ordered after filtering")

    return SupervisedDataset(
        X=X,
        y=y,
        feature_columns=available,
        target_name=target_name,
        timestamps=timestamps,
    )


# ---------------------------------------------------------------------------
# Cross-validation splitting
# ---------------------------------------------------------------------------

def walk_forward_splits(
    dataset: SupervisedDataset,
    n_folds: int = 6,
    gap_steps: int = 96,
) -> list[dict]:
    """Generate expanding-window walk-forward cross-validation fold splits.

    Walk-forward CV is the correct evaluation strategy for time-series forecasting
    because it mimics the real deployment scenario: the model is always trained on
    data that precedes the evaluation period.  A random split would allow 'future'
    data to appear in the training set, which would give artificially optimistic
    results (data leakage).

    Split structure for each fold i (0-indexed):
    ┌────────────────────────────────────────────────────────────┐
    │   TRAIN (expanding)  │ GAP │   VAL  │ GAP │   TEST         │
    └────────────────────────────────────────────────────────────┘
    → Time →

    Key design choices:
    - The initial 55% of the time range is used as the minimum training window.
    - The remaining 45% is divided evenly across n_folds for testing.
    - A 1-day gap (gap_steps × 15 min = 96 × 15 = 1440 min) between train and
      test prevents same-day leakage through lag features.
    - Boundaries are computed from timestamps, not DataFrame indices.  This makes
      splits robust when data has been filtered (e.g. quality-filtered rows removed).
    - The training set grows with each fold (expanding window), which means later
      folds benefit from more training data — consistent with production deployment
      where retraining happens on growing historical data.

    Args:
        dataset: SupervisedDataset with timestamps aligned to X and y.
        n_folds: Number of CV folds (6 gives Jul-Dec 2025 coverage ~5.5 weeks each).
        gap_steps: Gap between train end and test start in 15-min steps.
            96 steps = 24 hours = 1 day, which prevents lag-feature leakage.

    Returns:
        List of fold dictionaries, each containing:
            fold: 1-indexed fold number.
            train_idx: List of integer positions into dataset.X for training.
            val_idx: List of integer positions for validation (early stopping).
            test_idx: List of integer positions for evaluation.
            train_size, val_size, test_size: Convenience counts.
            train_range, test_range: Human-readable timestamp strings.
    """
    n = len(dataset.X)
    timestamps = dataset.timestamps

    # Compute split boundaries from actual timestamps for gap-awareness.
    ts_min = timestamps.iloc[0]
    ts_max = timestamps.iloc[-1]
    total_duration = ts_max - ts_min

    # Each fold's test window covers ~(45%/n_folds) of the total time range.
    test_duration = total_duration * 0.45 / n_folds
    gap_duration = pd.Timedelta(minutes=gap_steps * 15)
    # Validation window = half of test window size (used for early stopping).
    val_duration = test_duration * 0.5

    folds = []
    for fold_idx in range(n_folds):
        # The test window slides forward in time with each fold.
        # fold 0 tests the first 1/n_folds of the holdout period,
        # fold 1 tests the second 1/n_folds, etc.
        test_start_ts = ts_min + total_duration * 0.55 + fold_idx * test_duration
        test_end_ts = test_start_ts + test_duration

        # Validation sits immediately before the test window (with a gap).
        val_end_ts = test_start_ts - gap_duration
        val_start_ts = val_end_ts - val_duration

        # Training is everything chronologically before the validation window.
        # As fold_idx increases, train_end_ts also increases → expanding window.
        train_end_ts = val_start_ts

        # Convert timestamp boundaries back to integer indices.
        train_idx = [i for i in range(n) if timestamps.iloc[i] < train_end_ts]
        val_idx = [i for i in range(n) if val_start_ts <= timestamps.iloc[i] < val_end_ts]
        test_idx = [i for i in range(n) if test_start_ts <= timestamps.iloc[i] < test_end_ts]

        # Skip folds where any split is too small to be statistically meaningful.
        if len(train_idx) < 200 or len(val_idx) < 50 or len(test_idx) < 50:
            print(f"[warn] Fold {fold_idx + 1} skipped: train={len(train_idx)}, "
                  f"val={len(val_idx)}, test={len(test_idx)}")
            continue

        folds.append({
            "fold": fold_idx + 1,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
            "train_range": f"{timestamps.iloc[train_idx[0]]} .. {timestamps.iloc[train_idx[-1]]}",
            "test_range": f"{timestamps.iloc[test_idx[0]]} .. {timestamps.iloc[test_idx[-1]]}",
        })

    return folds
