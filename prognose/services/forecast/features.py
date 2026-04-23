"""Live-data feature engineering for real-time occupancy forecasting.

This module handles feature construction for the *inference* path (live prediction),
as opposed to features_excel.py which handles training.  The key difference is the
data source:

  - Training (features_excel.py): 15-minute Excel snapshots for the full year 2025.
  - Inference (this file): 1-minute live sensor readings from the database.

Because the two sources have different time resolutions and column names, there is a
structural mismatch between training features and inference features.  A mapping layer
(LIVE_TO_EXCEL dict) in main.py bridges the gap by renaming and filling missing columns
before passing data to the GBDT model.

Feature groups produced:
  1. Occupancy lags (1-min steps): lag_1 .. lag_60 at various intervals
  2. Rolling statistics: rolling mean and std at windows 5, 15, 60 min
  3. First-order differences at 1, 5, 15 min
  4. Cyclic temporal encodings (sine/cosine) for minute-of-day and day-of-week
  5. Calendar event features: event_active, event_impact_sum
  6. Lecture features: active counts, starts/ends, net pull, quality score, flags

Feature set version is "short_term_v2" (stored in every model bundle's metadata).

Key functions:
    build_feature_frame: Main entry point — takes raw history + context, returns
        a feature-engineered DataFrame (1-min resolution, NaN-free).
    build_supervised_dataset: Wrap build_feature_frame output into X/y matrices
        for multi-step training (used by the legacy TF-MLP path).
    build_inference_vector: Extract the single most-recent complete feature row
        for making a live prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Lag offsets in minutes used for autoregressive features.
# These are in 1-minute steps (unlike features_excel.py which uses 15-min steps).
LAG_STEPS = [1, 2, 3, 5, 10, 15, 30, 60]

# Rolling window sizes in minutes for mean and std features.
ROLL_WINDOWS = [5, 15, 60]

# Default expected attendance per lecture, used when the actual pull value is
# not available in the lecture metadata.
DEFAULT_LECTURE_AVG_ATTENDANCE = 20.0

# Version tag embedded in saved model bundles for traceability.
# Changing features requires bumping this version and retraining.
FEATURE_SET_VERSION = "short_term_v2"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SupervisedDataset:
    """Multi-output supervised dataset used by the legacy TF-MLP training path.

    Unlike the GBDT SupervisedDataset (in features_excel.py), this variant supports
    multi-step ahead targets: y contains one column per forecast horizon step.

    Attributes:
        X: Feature matrix (n_samples × n_features).
        y: Target matrix (n_samples × horizon), one column per future step.
        feature_columns: Column names of X.
        target_columns: Column names of y (e.g. ["y_tplus_1", ..., "y_tplus_60"]).
    """
    X: pd.DataFrame
    y: pd.DataFrame
    feature_columns: list[str]
    target_columns: list[str]


# ---------------------------------------------------------------------------
# Internal helper functions
# ---------------------------------------------------------------------------

def _normalize_ts(value: datetime) -> datetime:
    """Ensure a datetime is UTC-aware.

    Attaches UTC timezone if the datetime is naive; converts to UTC if it has
    a different timezone.  This prevents comparison errors when mixing
    timezone-aware and timezone-naive timestamps.

    Args:
        value: Any Python datetime object.

    Returns:
        UTC-aware datetime.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _to_history_frame(history_df: pd.DataFrame) -> pd.DataFrame:
    """Resample raw occupancy history to a regular 1-minute grid.

    Sensor readings may arrive at irregular intervals (network latency, buffering).
    This function standardises them to a 1-minute grid via resampling and linear
    interpolation so that lag and rolling features have consistent window sizes.

    Quality columns (quality_score, quality_flag_count, utilization) are also
    normalised here as they are needed downstream for data quality features.

    Args:
        history_df: Raw DataFrame with at minimum 'timestamp' and 'occupancy'.
            Optional columns: 'quality_score', 'quality_flags', 'utilization'.

    Returns:
        DataFrame indexed by UTC minute timestamps with columns:
            occupancy, utilization, quality_score, quality_flag_count.
        Returns an empty DataFrame if history_df is empty.
    """
    if history_df.empty:
        return pd.DataFrame()

    df = history_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")

    # Convert quality_flags (a list of strings per row) to a count.
    # More flags = lower measurement confidence.
    if "quality_flags" in df.columns:
        df["quality_flag_count"] = df["quality_flags"].apply(
            lambda flags: float(len(flags)) if isinstance(flags, list) else 0.0
        )
    else:
        df["quality_flag_count"] = 0.0

    # Default quality columns if not present in the input.
    if "quality_score" not in df.columns:
        df["quality_score"] = 1.0
    if "utilization" not in df.columns:
        df["utilization"] = 0.0

    # Pivot to minute-indexed DataFrame and resample to fill gaps.
    df = df.set_index("timestamp")
    minute = (
        df[["occupancy", "utilization", "quality_score", "quality_flag_count"]]
        .resample("1min")           # Create a regular 1-min grid
        .mean(numeric_only=True)    # Average if multiple readings per minute
        .sort_index()
    )

    # Linear interpolation fills short sensor gaps (e.g. 5-min dropouts).
    # Forward/backward fill is used for quality metadata.
    minute["occupancy"] = minute["occupancy"].interpolate(limit_direction="both")
    minute["utilization"] = minute["utilization"].interpolate(limit_direction="both")
    minute["quality_score"] = minute["quality_score"].ffill().bfill().fillna(1.0)
    minute["quality_flag_count"] = minute["quality_flag_count"].fillna(0.0)
    return minute


def _build_event_features(index: pd.DatetimeIndex, events_df: pd.DataFrame | None) -> pd.DataFrame:
    """Build binary event coverage and cumulative impact features.

    For each timestamp in `index`, this function checks whether a calendar event
    is active and sums the expected attendance impact of overlapping events.
    Events are things like special library sessions, university open days, etc.

    Args:
        index: DatetimeIndex of the feature frame (UTC, 1-min resolution).
        events_df: DataFrame with columns: starts_at, ends_at, expected_impact.
            Can be None or empty if no events are available.

    Returns:
        DataFrame with columns:
            event_active: 1.0 if any event covers this timestamp, else 0.0.
            event_impact_sum: Sum of expected_impact for all active events.
    """
    event_active = np.zeros(len(index), dtype=float)
    event_impact_sum = np.zeros(len(index), dtype=float)

    if events_df is None or events_df.empty or len(index) == 0:
        # No event data available — return zeros so downstream code still works.
        return pd.DataFrame(
            {"event_active": event_active, "event_impact_sum": event_impact_sum},
            index=index,
        )

    ev = events_df.copy()
    ev["starts_at"] = pd.to_datetime(ev["starts_at"], utc=True)
    ev["ends_at"] = pd.to_datetime(ev["ends_at"], utc=True)
    ev["expected_impact"] = ev.get("expected_impact", 0.0).fillna(0.0)

    for row in ev.itertuples(index=False):
        start = getattr(row, "starts_at")
        end = getattr(row, "ends_at")
        impact = float(getattr(row, "expected_impact", 0.0) or 0.0)
        # Boolean mask selects all 1-minute index positions that fall within the event.
        mask = (index >= start) & (index <= end)
        event_active[mask] = 1.0
        event_impact_sum[mask] += impact  # Additive: overlapping events stack

    return pd.DataFrame(
        {"event_active": event_active, "event_impact_sum": event_impact_sum},
        index=index,
    )


def _build_lecture_features(
    index: pd.DatetimeIndex,
    lecture_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build lecture activity features for each minute in the index.

    Lecture data captures how the DHBW Mannheim course schedule affects
    library occupancy:
    - When many lectures are active: students are in class → lower occupancy.
    - When lectures end: students leave class and may visit the library → spike.
    - 'Heavy' periods (≥3 parallel lectures) have a stronger pull effect.

    The raw lecture data arrives at irregular timestamps, so it is resampled to
    the 1-min grid and interpolated to fill gaps.

    Args:
        index: DatetimeIndex for alignment (UTC, 1-min resolution).
        lecture_df: DataFrame with columns: timestamp, active_lectures,
            starts_next_60m, ends_next_60m, quality_score, metadata (dict).
            Can be None or empty to produce zero-filled features.

    Returns:
        DataFrame indexed like `index` with lecture feature columns:
            lecture_count_now, lecture_starts_next_60m, lecture_ends_next_60m,
            lecture_quality_score, lecture_heavy_now, lecture_heavy_post_60m,
            lecture_pull_regular, lecture_bib_bonus, lecture_net_pull,
            lecture_count_roll_60, lecture_low_period_flag.
    """

    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Convert to float safely, returning default on error or infinity."""
        try:
            out = float(value)
        except Exception:
            return default
        if not np.isfinite(out):
            return default
        return out

    def _metadata_float(metadata: Any, key: str, default: float = 0.0) -> float:
        """Extract a numeric field from a metadata dict, returning default if absent."""
        if not isinstance(metadata, dict):
            return default
        return _safe_float(metadata.get(key), default)

    # Initialise all feature arrays to zero (used when lecture data is unavailable).
    lecture_count_now = np.zeros(len(index), dtype=float)
    lecture_starts_next_60m = np.zeros(len(index), dtype=float)
    lecture_ends_next_60m = np.zeros(len(index), dtype=float)
    lecture_quality = np.ones(len(index), dtype=float)
    lecture_heavy_now = np.zeros(len(index), dtype=float)
    lecture_heavy_post_60m = np.zeros(len(index), dtype=float)
    lecture_pull_regular = np.zeros(len(index), dtype=float)
    lecture_bib_bonus = np.zeros(len(index), dtype=float)
    lecture_net_pull = np.zeros(len(index), dtype=float)

    if lecture_df is None or lecture_df.empty or len(index) == 0:
        # Return zero-filled frame when no lecture data is available.
        frame = pd.DataFrame(
            {
                "lecture_count_now": lecture_count_now,
                "lecture_starts_next_60m": lecture_starts_next_60m,
                "lecture_ends_next_60m": lecture_ends_next_60m,
                "lecture_quality_score": lecture_quality,
                "lecture_heavy_now": lecture_heavy_now,
                "lecture_heavy_post_60m": lecture_heavy_post_60m,
                "lecture_pull_regular": lecture_pull_regular,
                "lecture_bib_bonus": lecture_bib_bonus,
                "lecture_net_pull": lecture_net_pull,
            },
            index=index,
        )
        # Derived aggregated feature: rolling mean of active lectures over 60 min.
        frame["lecture_count_roll_60"] = (
            pd.Series(lecture_count_now, index=index)
            .rolling(window=60, min_periods=1)
            .mean()
        )
        # Flag: 1 if this is a 'quiet lecture period' (fewer than 1 lecture on avg
        # over the last hour).  Useful for distinguishing study vs. lecture time.
        frame["lecture_low_period_flag"] = (frame["lecture_count_roll_60"] < 1.0).astype(float)
        return frame

    df = lecture_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")
    if "metadata" not in df.columns:
        df["metadata"] = [{} for _ in range(len(df))]

    # Ensure all expected columns exist (with sensible defaults if missing).
    for col in ["active_lectures", "starts_next_60m", "ends_next_60m", "quality_score"]:
        if col not in df.columns:
            df[col] = 0.0 if col != "quality_score" else 1.0

    # Extract sub-fields from the JSON metadata column.
    # 'heavy_active_lectures': number of parallel lectures above the 'heavy' threshold.
    df["lecture_heavy_now"] = df["metadata"].apply(
        lambda value: _metadata_float(value, "heavy_active_lectures", 0.0)
    )
    # 'heavy_ended_last_60m': heavy lectures that ended recently → library influx expected.
    df["lecture_heavy_post_60m"] = df["metadata"].apply(
        lambda value: _metadata_float(value, "heavy_ended_last_60m", 0.0)
    )
    # 'lecture_pull_regular': estimated number of students expected to come to the
    # library from currently active lectures (lectures × avg attendance per lecture).
    df["lecture_pull_regular"] = [
        _metadata_float(
            metadata,
            "lecture_pull_regular",
            _safe_float(active, 0.0) * DEFAULT_LECTURE_AVG_ATTENDANCE,
        )
        for metadata, active in zip(df["metadata"], df["active_lectures"])
    ]
    # 'bib_bonus': extra occupancy expected during library-specific events / activities.
    df["lecture_bib_bonus"] = df["metadata"].apply(
        lambda value: _metadata_float(value, "heavy_bib_bonus", 0.0)
    )
    # 'net_pull': net occupancy pull = lecture pull minus bonus students already counted.
    df["lecture_net_pull"] = [
        _metadata_float(
            metadata,
            "lecture_net_pull",
            _safe_float(pull, 0.0) - _safe_float(bonus, 0.0),
        )
        for metadata, pull, bonus in zip(
            df["metadata"], df["lecture_pull_regular"], df["lecture_bib_bonus"]
        )
    ]

    # Resample to 1-min grid and align to the feature index via reindex.
    minute = (
        df[[
            "active_lectures", "starts_next_60m", "ends_next_60m", "quality_score",
            "lecture_heavy_now", "lecture_heavy_post_60m",
            "lecture_pull_regular", "lecture_bib_bonus", "lecture_net_pull",
        ]]
        .resample("1min")
        .mean(numeric_only=True)
        .reindex(index)
    )

    # Fill gaps: numeric lecture counts by linear interpolation, quality by forward fill.
    for col in ["active_lectures", "starts_next_60m", "ends_next_60m",
                "lecture_heavy_now", "lecture_heavy_post_60m",
                "lecture_pull_regular", "lecture_bib_bonus", "lecture_net_pull"]:
        minute[col] = minute[col].interpolate(limit_direction="both").fillna(0.0)
    minute["quality_score"] = minute["quality_score"].ffill().bfill().fillna(0.8)

    frame = pd.DataFrame(
        {
            "lecture_count_now": minute["active_lectures"].values.astype(float),
            "lecture_starts_next_60m": minute["starts_next_60m"].values.astype(float),
            "lecture_ends_next_60m": minute["ends_next_60m"].values.astype(float),
            "lecture_quality_score": minute["quality_score"].values.astype(float),
            "lecture_heavy_now": minute["lecture_heavy_now"].values.astype(float),
            "lecture_heavy_post_60m": minute["lecture_heavy_post_60m"].values.astype(float),
            "lecture_pull_regular": minute["lecture_pull_regular"].values.astype(float),
            "lecture_bib_bonus": minute["lecture_bib_bonus"].values.astype(float),
            "lecture_net_pull": minute["lecture_net_pull"].values.astype(float),
        },
        index=index,
    )

    # Rolling 60-min mean: captures sustained lecture load vs. momentary spikes.
    frame["lecture_count_roll_60"] = (
        frame["lecture_count_now"].rolling(window=60, min_periods=1).mean()
    )
    # Low-period flag: 1 when the hourly-average active lecture count is below 1.
    # This signals 'study time' (no lectures pulling students away from the library).
    frame["lecture_low_period_flag"] = (
        (frame["lecture_count_roll_60"] < 1.0).astype(float)
    )
    return frame


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_feature_frame(
    history_df: pd.DataFrame,
    events_df: pd.DataFrame | None = None,
    lecture_df: pd.DataFrame | None = None,
    use_calendar_features: bool = True,
    include_lecture_impact: bool = True,
) -> pd.DataFrame:
    """Build the complete 1-minute feature frame from raw history and context data.

    This is the primary entry point for both training (legacy TF path) and
    inference.  It takes raw sensor history and optional context signals
    (calendar events, lecture schedules) and returns a clean, NaN-free feature
    DataFrame ready for model input.

    Feature construction pipeline:
      1. Resample history to 1-min grid (via _to_history_frame).
      2. Compute autoregressive lag features (LAG_STEPS).
      3. Compute rolling mean/std (ROLL_WINDOWS).
      4. Compute first-order differences (momentum).
      5. Compute cyclic temporal encodings (sine/cosine).
      6. Optionally join calendar event features.
      7. Optionally join lecture activity features.
      8. Drop NaN rows (caused by lags at the beginning of the series).

    Args:
        history_df: Raw occupancy history with 'timestamp' and 'occupancy' columns.
            Can contain irregular timestamps — will be resampled to 1-min grid.
        events_df: Optional calendar events with starts_at, ends_at, expected_impact.
        lecture_df: Optional DHBW lecture activity data (see _build_lecture_features).
        use_calendar_features: If False, event features are zero-filled (useful for
            testing or when event data is unavailable).
        include_lecture_impact: If False, lecture features are zero-filled.

    Returns:
        Feature DataFrame indexed by UTC minute timestamps, NaN-free.
        Returns an empty DataFrame if history_df is empty.
    """
    # Step 1: Resample raw history to a regular 1-min grid.
    base = _to_history_frame(history_df)
    if base.empty:
        return pd.DataFrame()

    feature_df = base.copy()

    # Step 2: Autoregressive lag features — past occupancy at various intervals.
    # Lags are the single most predictive feature group for occupancy forecasting
    # because the series is strongly autocorrelated.
    for lag in LAG_STEPS:
        feature_df[f"occupancy_lag_{lag}"] = feature_df["occupancy"].shift(lag)

    # Step 3: Rolling statistics — smooth the signal and capture volatility.
    for window in ROLL_WINDOWS:
        roll = feature_df["occupancy"].rolling(window=window, min_periods=window)
        feature_df[f"occupancy_roll_mean_{window}"] = roll.mean()
        feature_df[f"occupancy_roll_std_{window}"] = roll.std(ddof=0)

    # Step 4: First-order differences — rate of change (is occupancy rising/falling?).
    feature_df["occupancy_diff_1"] = feature_df["occupancy"].diff(1)   # vs. 1 min ago
    feature_df["occupancy_diff_5"] = feature_df["occupancy"].diff(5)   # vs. 5 min ago
    feature_df["occupancy_diff_15"] = feature_df["occupancy"].diff(15) # vs. 15 min ago

    # Step 5: Cyclic temporal encodings.
    # Using sin/cos makes time periodic so that the feature space wraps around at
    # day/week boundaries (e.g. 23:59 → 00:00 is a small step, not a large one).
    minute_of_day = feature_df.index.hour * 60 + feature_df.index.minute
    day_of_week = feature_df.index.dayofweek
    feature_df["minute_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    feature_df["minute_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    feature_df["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    feature_df["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)

    # Step 6: Calendar event features.
    if use_calendar_features:
        event_df = _build_event_features(feature_df.index, events_df)
        feature_df = feature_df.join(event_df)
    else:
        feature_df["event_active"] = 0.0
        feature_df["event_impact_sum"] = 0.0

    # Step 7: Lecture activity features.
    lecture_features = _build_lecture_features(
        feature_df.index,
        lecture_df if include_lecture_impact else None,
    )
    feature_df = feature_df.join(lecture_features)

    # Step 8: Drop rows with any NaN (first LAG_STEPS rows will be NaN from lags).
    feature_df = feature_df.dropna()
    return feature_df


def build_supervised_dataset(feature_df: pd.DataFrame, horizon: int) -> SupervisedDataset:
    """Create a multi-step supervised dataset for the legacy TF-MLP training path.

    For each row, creates `horizon` future targets (y_tplus_1 through y_tplus_N).
    This multi-step formulation is used by the old TensorFlow MLP which predicted
    all steps simultaneously.  The new GBDT model uses a single-step target instead
    (defined in features_excel.py).

    Args:
        feature_df: Output of build_feature_frame (1-min, NaN-free).
        horizon: Number of future 1-minute steps to create targets for.

    Returns:
        SupervisedDataset with X (features) and y (multi-step targets).

    Raises:
        ValueError: If horizon < 1 or feature_df is empty.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if feature_df.empty:
        raise ValueError("feature_df is empty")

    dataset = feature_df.copy()
    target_columns: list[str] = []
    for step in range(1, horizon + 1):
        # Shift occupancy forward by `step` positions to create target at t+step.
        col = f"y_tplus_{step}"
        dataset[col] = dataset["occupancy"].shift(-step)
        target_columns.append(col)

    # Drop rows where any target is NaN (last `horizon` rows have no future).
    dataset = dataset.dropna()
    if dataset.empty:
        raise ValueError("not enough rows after target alignment")

    feature_columns = [col for col in dataset.columns if col not in target_columns]
    X = dataset[feature_columns]
    y = dataset[target_columns]
    return SupervisedDataset(
        X=X, y=y,
        feature_columns=feature_columns,
        target_columns=target_columns,
    )


def build_inference_vector(
    feature_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, datetime]:
    """Extract the single most-recent complete feature row for live inference.

    At inference time we only need one row: the latest available feature snapshot.
    This function finds the last row where all required features are non-NaN and
    returns it together with its timestamp.

    Args:
        feature_df: Output of build_feature_frame (1-min, NaN-free in most rows).
        feature_columns: Exact list of feature names the model expects, in order.

    Returns:
        Tuple of (single-row DataFrame, UTC datetime of that row).

    Raises:
        ValueError: If feature_df is empty, required columns are missing, or no
            complete (NaN-free) row exists.
    """
    if feature_df.empty:
        raise ValueError("feature_df is empty")

    missing = [col for col in feature_columns if col not in feature_df.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")

    # Select only the required columns and take the last NaN-free row.
    latest = feature_df[feature_columns].dropna().tail(1)
    if latest.empty:
        raise ValueError("no complete feature row available for inference")

    # Convert the index timestamp to a normalised UTC datetime.
    ts = _normalize_ts(latest.index[-1].to_pydatetime())
    return latest, ts
