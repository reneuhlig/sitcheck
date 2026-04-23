#!/usr/bin/env python3
"""Prepare training data from KI_Projekt_Daten_einJahr.xlsx + DHBW lecture profiles.

This script is the first step of the training pipeline. It takes raw occupancy
measurements from an Excel file (one year of 15-minute sensor readings) and
enriches them with two kinds of engineered features:
  1. Time-based features: cyclic encodings, lag values, rolling statistics
  2. Lecture-proxy features: projected from a weekly DHBW lecture activity profile

The output is a single Parquet file that `train_gbdt.py` reads directly. Keeping
data preparation in a separate script makes the training pipeline reproducible —
you can regenerate training data independently of model code.

Key functions:
    load_excel: Load and validate the raw Excel dataset.
    extract_lecture_profile: Build a (weekday, hour) lookup table from the DB.
    build_training_features: Combine Excel data + lecture profile into 40+ features.
    print_quality_report: Print a human-readable summary for data quality checks.
    main: CLI entry point wiring all steps together.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default file paths (resolved relative to this script's location in the repo)
# ---------------------------------------------------------------------------
EXCEL_DEFAULT = str(Path(__file__).resolve().parents[2] / "KI_Projekt_Daten_einJahr.xlsx")
DB_DEFAULT = str(Path(__file__).resolve().parents[2] / "runtime_local.db")
OUTPUT_DEFAULT = str(Path(__file__).resolve().parents[2] / "training_data.parquet")

# Physical capacity of the library zone being modelled.
# This constant must match the value in services/forecast/features_excel.py
# so that target clipping in training is consistent with inference bounds.
CAPACITY_TOTAL = 84


def load_excel(path: str) -> pd.DataFrame:
    """Load and validate the raw Excel training dataset.

    Reads the one-year occupancy file, enforces timezone-aware UTC timestamps,
    sorts chronologically, and asserts the minimum expected columns and row count.
    Failing fast here prevents silent bugs in downstream feature engineering.

    Args:
        path: Absolute or relative path to the Excel file (.xlsx).

    Returns:
        Sorted DataFrame with columns including 'timestamp' and 'occupancy'.

    Raises:
        AssertionError: If required columns are missing or row count is too low.
    """
    df = pd.read_excel(path, engine="openpyxl")
    print(f"[info] loaded Excel: {len(df)} rows, columns: {list(df.columns)}")

    # Ensure timestamps are UTC-aware so downstream merges with DB data are
    # timezone-consistent.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Basic validation — fail early if the file structure is unexpected.
    assert "occupancy" in df.columns, "Missing 'occupancy' column"
    assert "capacity_effective" in df.columns, "Missing 'capacity_effective' column"
    assert len(df) > 1000, f"Too few rows: {len(df)}"

    print(f"[info] time range: {df['timestamp'].min()} .. {df['timestamp'].max()}")
    print(f"[info] occupancy: mean={df['occupancy'].mean():.1f}, "
          f"median={df['occupancy'].median():.1f}, "
          f"max={df['occupancy'].max()}, "
          f"zeros={int((df['occupancy'] == 0).sum())}")
    return df


def extract_lecture_profile(db_path: str) -> pd.DataFrame:
    """Extract an average weekly lecture activity profile from the SQLite database.

    DHBW lecture schedules are not known day-by-day at inference time, so we
    approximate them with a weekly average: for each (weekday, hour) combination,
    we compute the mean of how many lectures were active, starting, ending, or
    classified as 'heavy' (>= 3 parallel lectures). This profile is then used to
    inject lecture-awareness into the time-series feature set.

    If the database does not exist or the table is empty, a zero-filled profile is
    returned so training can still proceed without lecture features.

    Args:
        db_path: Path to the SQLite database containing the `lecture_activity` table.

    Returns:
        DataFrame indexed by (weekday, hour) with columns:
            lecture_density: mean active lecture count
            lecture_starts: mean lectures starting in next 60 min
            lecture_ends: mean lectures ending in next 60 min
            lecture_heavy: mean count of 'heavy' (high-density) lecture slots
    """
    if not Path(db_path).exists():
        print(f"[warn] DB not found: {db_path}, using zero lecture profile")
        # Build an all-zero profile covering every (weekday 0-6, hour 0-23) pair.
        idx = pd.MultiIndex.from_product(
            [range(7), range(24)], names=["weekday", "hour"]
        )
        return pd.DataFrame(
            {"lecture_density": 0.0, "lecture_starts": 0.0,
             "lecture_ends": 0.0, "lecture_heavy": 0.0},
            index=idx,
        )

    conn = sqlite3.connect(db_path)
    query = """
        SELECT ts, active_lectures, starts_next_60m, ends_next_60m, metadata
        FROM lecture_activity
        ORDER BY ts
    """
    ldf = pd.read_sql_query(query, conn)
    conn.close()

    if ldf.empty:
        print("[warn] lecture_activity table is empty, using zero profile")
        idx = pd.MultiIndex.from_product(
            [range(7), range(24)], names=["weekday", "hour"]
        )
        return pd.DataFrame(
            {"lecture_density": 0.0, "lecture_starts": 0.0,
             "lecture_ends": 0.0, "lecture_heavy": 0.0},
            index=idx,
        )

    ldf["timestamp"] = pd.to_datetime(ldf["ts"], utc=True)
    ldf["weekday"] = ldf["timestamp"].dt.weekday
    ldf["hour"] = ldf["timestamp"].dt.hour

    # The `heavy_active_lectures` count is stored as a JSON field inside
    # the `metadata` column rather than a dedicated column.
    def _get_heavy(meta_str: str) -> float:
        """Parse `heavy_active_lectures` from the metadata JSON blob."""
        try:
            meta = json.loads(meta_str) if isinstance(meta_str, str) else {}
            return float(meta.get("heavy_active_lectures", 0))
        except Exception:
            return 0.0

    ldf["heavy_active"] = ldf["metadata"].apply(_get_heavy)

    # Average over all historical occurrences for each (weekday, hour) bucket.
    # This gives us a 'typical week' profile that can be looked up by timestamp.
    profile = ldf.groupby(["weekday", "hour"]).agg(
        lecture_density=("active_lectures", "mean"),
        lecture_starts=("starts_next_60m", "mean"),
        lecture_ends=("ends_next_60m", "mean"),
        lecture_heavy=("heavy_active", "mean"),
    )

    print(f"[info] lecture profile: {len(profile)} (weekday, hour) entries, "
          f"avg density={profile['lecture_density'].mean():.1f}")
    return profile


def build_training_features(df: pd.DataFrame, lecture_profile: pd.DataFrame) -> pd.DataFrame:
    """Build the complete 40-feature training DataFrame from Excel data + lecture profile.

    Feature engineering is the most critical step for model quality. The 40 features
    are grouped into nine categories:

    1. Excel-native factors (7): pre-computed scores and flags from the original dataset
       (month factor, weekday factor, time-of-day factor, weather factor, bridge-day
       factor, efficiency score, effective capacity).
    2. Binary flags (5): bridge_day, winter_break, weather states, partial closure.
    3. Utilization (1): occupancy as a percentage of effective capacity.
    4. Temporal/cyclic (8): sine/cosine encodings for minute-of-day, day-of-week, and
       day-of-year. Cyclic encoding is used instead of raw integers so that, for example,
       Sunday (6) and Monday (0) appear numerically close to each other.
    5. Occupancy lags (8): past occupancy values at intervals of 1, 2, 3, 4, 8, and 16
       steps (each step = 15 min), plus 'same slot yesterday' (48 steps = 12 operating
       hours) and 'same slot last week' (288 steps = 6 working days × 12 h).
    6. Rolling statistics (5): rolling mean over windows of 1h, 2h, 4h; rolling std
       over 1h and 2h. These capture short-term momentum and volatility.
    7. Differences (2): first-order differences at 1 step and 4 steps (1 hour), which
       capture whether occupancy is rising or falling.
    8. Lecture proxies (4): values from the weekly profile, projected by (weekday, hour).
    9. Calendar + lecture real-time (from features_excel.py when full data is available).

    Rows containing NaN values (caused by lags at the beginning of the series) are
    dropped before saving.

    Args:
        df: Raw DataFrame from load_excel, with at minimum 'timestamp' and 'occupancy'.
        lecture_profile: (weekday, hour) lookup DataFrame from extract_lecture_profile.

    Returns:
        Feature DataFrame ready for train_gbdt.py, with NaN rows removed.
    """
    out = pd.DataFrame(index=df.index)
    out["timestamp"] = df["timestamp"]
    out["occupancy"] = df["occupancy"].astype(float)

    # -----------------------------------------------------------------------
    # 1. Excel-native factors — copied directly from the dataset.
    #    These were pre-computed by the study team and encode domain knowledge
    #    (e.g. f_weather encodes the expected weather impact on attendance).
    # -----------------------------------------------------------------------
    for col in ["f_month", "f_weekday", "f_tod", "f_weather", "f_bridge",
                 "efficiency", "capacity_effective"]:
        if col in df.columns:
            out[col] = df[col].astype(float)

    out["bridge_day"] = df["bridge_day"].astype(int) if "bridge_day" in df.columns else 0
    out["winter_break"] = df["winter_break"].astype(int) if "winter_break" in df.columns else 0

    # -----------------------------------------------------------------------
    # 2. Weather binary flags.
    #    One-hot encoding is preferred over a single ordinal because the model
    #    should treat 'rainy' and 'sunny' as independent signals, not as ordered
    #    values on a numeric scale.
    # -----------------------------------------------------------------------
    if "weather" in df.columns:
        out["weather_rainy"] = (df["weather"] == "rainy").astype(int)
        out["weather_sunny"] = (df["weather"] == "sunny").astype(int)
    else:
        out["weather_rainy"] = 0
        out["weather_sunny"] = 0

    # Partial closure events reduce available space and suppress attendance.
    if "event_note" in df.columns:
        out["is_partial_closure"] = df["event_note"].fillna("").str.contains(
            "partial_closure", case=False, na=False
        ).astype(int)
    else:
        out["is_partial_closure"] = 0

    # -----------------------------------------------------------------------
    # 3. Utilization percentage: occupancy / effective_capacity * 100.
    #    This normalised value helps the model generalise if capacity changes
    #    over time (e.g. sections are closed for refurbishment).
    # -----------------------------------------------------------------------
    if "utilization_pct" in df.columns:
        out["utilization_pct"] = df["utilization_pct"].astype(float)
    else:
        # Compute from first principles if not pre-supplied.
        cap = out["capacity_effective"].replace(0, np.nan)
        out["utilization_pct"] = (out["occupancy"] / cap * 100).fillna(0)

    # -----------------------------------------------------------------------
    # 4. Temporal / cyclic features.
    #    We encode time as sine/cosine pairs so the representation is continuous
    #    and periodic. For example, minute 1439 (23:59) and minute 0 (00:00) are
    #    encoded as nearly identical vectors, which a linear model or tree-split
    #    cannot achieve with raw integer minutes.
    # -----------------------------------------------------------------------
    ts = df["timestamp"]
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute   # 0..1439
    day_of_week = ts.dt.dayofweek                      # 0=Monday .. 6=Sunday
    day_of_year = ts.dt.dayofyear                      # 1..365

    # Cyclic sine/cosine pairs for each temporal period.
    out["minute_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    out["minute_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    out["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)
    out["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.0)
    out["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.0)

    # Also include raw hour and weekday flag as non-cyclic features for the
    # tree model — decision trees can use these as direct split thresholds.
    out["hour_of_day"] = ts.dt.hour
    out["is_weekday"] = (day_of_week < 5).astype(int)   # 1 = Mon-Fri, 0 = Sat/Sun

    # -----------------------------------------------------------------------
    # 5. Occupancy lag features (autoregressive inputs).
    #    Lags are the most predictive feature group because occupancy is strongly
    #    autocorrelated: the library tends to be full when it was recently full.
    #    Each step = 15 minutes (the measurement interval).
    # -----------------------------------------------------------------------
    for lag in [1, 2, 3, 4, 8, 16]:
        # lag_1  =  15 min ago, lag_4  = 60 min ago (= same as horizon target),
        # lag_8  = 120 min ago, lag_16 = 240 min ago.
        out[f"occupancy_lag_{lag}"] = out["occupancy"].shift(lag)

    # "Same time yesterday" proxy.
    # The library operates ~12 operating hours per day (approx. 10:00-22:00),
    # so 48 steps of 15 min = 12 h maps to the equivalent slot in the previous
    # working day. This captures the day-over-day seasonal pattern.
    out["occupancy_lag_day"] = out["occupancy"].shift(48)

    # "Same time last week" proxy.
    # 6 working days (Mon-Sat) × 48 steps/day = 288 steps = 72 calendar hours.
    # This captures weekly seasonality (e.g. Monday mornings are consistently quiet).
    out["occupancy_lag_week"] = out["occupancy"].shift(288)

    # -----------------------------------------------------------------------
    # 6. Rolling statistics: momentum and volatility indicators.
    #    Rolling means smooth out noise and represent 'average recent occupancy'.
    #    Rolling std captures how variable (volatile) occupancy has been recently.
    # -----------------------------------------------------------------------
    for w in [4, 8, 16]:
        # windows: 4=1h, 8=2h, 16=4h
        roll = out["occupancy"].rolling(window=w, min_periods=1)
        out[f"occupancy_roll_mean_{w}"] = roll.mean()
        if w <= 8:
            # Standard deviation only for shorter windows; longer windows add
            # noise more than signal for this target.
            out[f"occupancy_roll_std_{w}"] = roll.std(ddof=0).fillna(0)

    # -----------------------------------------------------------------------
    # 7. First-order differences: rate of change of occupancy.
    #    Positive diff_1 means occupancy is rising; negative means falling.
    #    diff_4 captures the 1-hour trend, which is more stable than diff_1.
    # -----------------------------------------------------------------------
    out["occupancy_diff_1"] = out["occupancy"].diff(1)   # change vs. 15 min ago
    out["occupancy_diff_4"] = out["occupancy"].diff(4)   # change vs. 60 min ago

    # -----------------------------------------------------------------------
    # 8. Lecture proxy features: project the weekly lecture profile onto each
    #    row by looking up (weekday, hour).
    #    These capture DHBW-specific patterns: occupancy spikes after lectures
    #    end (students move to the library) and dips while lectures are ongoing.
    # -----------------------------------------------------------------------
    weekday = ts.dt.weekday
    hour = ts.dt.hour

    # Pre-allocate numpy arrays for speed (avoid per-row DataFrame writes).
    lecture_density = np.zeros(len(df))
    lecture_starts = np.zeros(len(df))
    lecture_ends = np.zeros(len(df))
    lecture_heavy = np.zeros(len(df))

    for i, (wd, h) in enumerate(zip(weekday, hour)):
        if (wd, h) in lecture_profile.index:
            row = lecture_profile.loc[(wd, h)]
            lecture_density[i] = row["lecture_density"]
            lecture_starts[i] = row["lecture_starts"]
            lecture_ends[i] = row["lecture_ends"]
            lecture_heavy[i] = row["lecture_heavy"]

    out["lecture_density_proxy"] = lecture_density
    out["lecture_starts_proxy"] = lecture_starts
    out["lecture_ends_proxy"] = lecture_ends
    out["lecture_heavy_proxy"] = lecture_heavy

    # -----------------------------------------------------------------------
    # Drop rows where any lag or rolling feature is NaN.
    # These occur at the beginning of the series where there is not enough
    # history to compute a lag or window. Keeping NaN rows would corrupt
    # the model with silent missing-value substitutions.
    # -----------------------------------------------------------------------
    out = out.dropna().reset_index(drop=True)
    print(f"[info] features built: {len(out)} rows, {len(out.columns)} columns")
    return out


def print_quality_report(df: pd.DataFrame) -> None:
    """Print a human-readable data quality summary to stdout.

    Used as a sanity check after feature engineering to catch obvious problems
    (extreme values, unexpectedly many zeros, large time gaps) before training.

    Args:
        df: The fully engineered training DataFrame.
    """
    print("\n=== DATA QUALITY REPORT ===")
    print(f"Rows: {len(df)}")
    print(f"Time range: {df['timestamp'].min()} .. {df['timestamp'].max()}")
    print(f"Columns: {len(df.columns)}")
    print(f"\nOccupancy stats:")
    occ = df["occupancy"]
    print(f"  mean={occ.mean():.1f}, std={occ.std():.1f}, "
          f"min={occ.min():.0f}, max={occ.max():.0f}")
    print(f"  zeros: {int((occ == 0).sum())} ({(occ == 0).mean()*100:.1f}%)")

    # Q99 helps spot sensor outliers without being skewed by extreme anomalies.
    q99 = occ.quantile(0.99)
    print(f"  Q99={q99:.1f}")

    # Check for unexpected time gaps (e.g. sensor downtime, missing data).
    # Expected interval between consecutive readings is 15 minutes (900 seconds).
    ts_diff = df["timestamp"].diff().dt.total_seconds().dropna()
    expected = 15 * 60  # 900 seconds
    gaps = ts_diff[ts_diff > expected * 1.5]  # flag gaps > 22.5 min
    print(f"\nTime gaps (>22.5 min): {len(gaps)}")
    if len(gaps) > 0 and len(gaps) <= 10:
        for idx in gaps.index[:10]:
            print(f"  {df['timestamp'].iloc[idx-1]} -> {df['timestamp'].iloc[idx]} "
                  f"({ts_diff[idx]/60:.0f} min)")

    print(f"\nFeature columns: {[c for c in df.columns if c not in ['timestamp', 'occupancy']]}")
    print("=== END REPORT ===\n")


def main() -> int:
    """CLI entry point: orchestrate the full data preparation pipeline.

    The pipeline runs in five sequential steps:
      1. Load raw Excel data.
      2. Extract lecture activity profile from the SQLite database.
      3. Engineer all features by joining Excel data + lecture profile.
      4. Print a quality report for human inspection.
      5. Save the result as a Parquet file (columnar format, fast to read).

    Returns:
        0 on success (used as shell exit code via SystemExit).
    """
    parser = argparse.ArgumentParser(description="Prepare training data from Excel + lectures")
    parser.add_argument("--excel", default=EXCEL_DEFAULT, help="Path to Excel file")
    parser.add_argument("--db", default=DB_DEFAULT, help="Path to SQLite database")
    parser.add_argument("--output", default=OUTPUT_DEFAULT, help="Output Parquet path")
    parser.add_argument("--report-only", action="store_true",
                        help="Only print quality report, do not write output file")
    args = parser.parse_args()

    # Step 1: Load and validate the raw Excel data.
    df = load_excel(args.excel)

    # Step 2: Build the weekly lecture activity profile from the DB.
    lecture_profile = extract_lecture_profile(args.db)

    # Step 3: Combine both sources into the engineered feature matrix.
    training_df = build_training_features(df, lecture_profile)

    # Step 4: Print quality report so the user can spot issues before training.
    print_quality_report(training_df)

    if args.report_only:
        # Useful for CI checks or quick validation without writing files.
        return 0

    # Step 5: Persist to Parquet for fast, typed reads by train_gbdt.py.
    training_df.to_parquet(args.output, index=False)
    print(f"[done] saved {len(training_df)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
