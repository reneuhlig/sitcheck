"""Weekly slot-based occupancy forecast (deterministic, no ML model).

This module produces a 7-day hourly occupancy outlook by aggregating patterns
from historical data and adjusting for known events and lecture schedules.

Why a separate weekly forecast?
  The GBDT model predicts a single 60-minute horizon.  For a 7-day view, users
  need coarser, pattern-based estimates — what the library typically looks like
  on a Tuesday at 14:00 in general, not a precise ML prediction.  This forecast
  is deterministic (no model, just pattern aggregation) and is displayed as the
  'weekly outlook' view in the dashboard.

Methodology for each future slot:
  1. Compute the historical mean occupancy for that (day-of-week, slot-of-day)
     combination from the last 28 days.
  2. Add a delta for calendar events (if any event overlaps this slot).
  3. Add a delta for lecture transitions (if lectures are starting or ending
     in this slot, based on the lecture_df activity data).
  4. Clip the final prediction to [0, capacity].
  5. Estimate an uncertainty band from the historical standard deviation.

Key functions:
    build_weekly_forecast: Main entry point. Returns a full weekly forecast dict
        with points, daily_summaries, and evidence lineage.
"""
from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd


# Product identifier embedded in evidence lineage records.
WEEKLY_PRODUCT = "weekly_slot"
# Feature set version for the weekly forecast (deterministic, no ML features).
WEEKLY_FEATURE_SET_VERSION = "weekly-slot-v1"
# Default number of historical days used to compute slot averages.
WEEKLY_DEFAULT_HISTORY_DAYS = 28


def _normalize_dt(value: datetime) -> datetime:
    """Ensure a datetime is UTC-aware; attach UTC if naive, convert otherwise."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _empty_lecture_slot_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Return a zero-filled lecture feature frame for when lecture data is absent."""
    return pd.DataFrame(
        {
            "active_lectures": np.zeros(len(index), dtype=float),
            "starts_next_60m": np.zeros(len(index), dtype=float),
            "ends_next_60m": np.zeros(len(index), dtype=float),
            "lecture_net_pull": np.zeros(len(index), dtype=float),
            "quality_score": np.ones(len(index), dtype=float),
        },
        index=index,
    )


def _history_to_slot_frame(history_df: pd.DataFrame, slot_minutes: int) -> pd.DataFrame:
    """Resample raw occupancy history to regular slot-sized intervals.

    Converts the raw 1-minute or irregular occupancy history into a regular
    grid of `slot_minutes`-wide bins by resampling and interpolating.

    Args:
        history_df: Raw history DataFrame with 'timestamp' and 'occupancy'.
        slot_minutes: Slot size in minutes (e.g. 60 = hourly bins).

    Returns:
        DataFrame indexed by slot timestamps with occupancy and quality columns.
    """
    if history_df.empty:
        return pd.DataFrame(
            columns=[
                "occupancy",
                "utilization",
                "quality_score",
                "quality_flag_count",
            ]
        )

    df = history_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "utilization" not in df.columns:
        df["utilization"] = 0.0
    if "quality_score" not in df.columns:
        df["quality_score"] = 1.0
    if "quality_flags" not in df.columns:
        df["quality_flags"] = [[] for _ in range(len(df))]
    df["quality_flag_count"] = df["quality_flags"].apply(
        lambda flags: float(len(flags)) if isinstance(flags, list) else 0.0
    )
    freq = f"{max(1, int(slot_minutes))}min"
    slot = (
        df.set_index("timestamp")[["occupancy", "utilization", "quality_score", "quality_flag_count"]]
        .resample(freq)
        .mean(numeric_only=True)
        .sort_index()
    )
    slot["occupancy"] = slot["occupancy"].interpolate(limit_direction="both")
    slot["utilization"] = slot["utilization"].interpolate(limit_direction="both").fillna(0.0)
    slot["quality_score"] = slot["quality_score"].ffill().bfill().fillna(0.8)
    slot["quality_flag_count"] = slot["quality_flag_count"].fillna(0.0)
    return slot.dropna(subset=["occupancy"])


def _events_to_slot_frame(
    *,
    index: pd.DatetimeIndex,
    events_df: pd.DataFrame | None,
    slot_minutes: int,
) -> pd.DataFrame:
    if len(index) == 0:
        return pd.DataFrame(columns=["event_active", "event_impact_sum"], index=index)

    frame = pd.DataFrame(
        {
            "event_active": np.zeros(len(index), dtype=float),
            "event_impact_sum": np.zeros(len(index), dtype=float),
        },
        index=index,
    )
    if events_df is None or events_df.empty:
        return frame

    freq = timedelta(minutes=max(1, int(slot_minutes)))
    events = events_df.copy()
    events["starts_at"] = pd.to_datetime(events["starts_at"], utc=True)
    events["ends_at"] = pd.to_datetime(events["ends_at"], utc=True)
    if "expected_impact" not in events.columns:
        events["expected_impact"] = 0.0
    events["expected_impact"] = pd.to_numeric(events["expected_impact"], errors="coerce").fillna(0.0)

    for row in events.itertuples(index=False):
        start = getattr(row, "starts_at")
        end = getattr(row, "ends_at")
        impact = float(getattr(row, "expected_impact", 0.0) or 0.0)
        slot_mask = (index + freq > start) & (index <= end)
        frame.loc[slot_mask, "event_active"] = 1.0
        frame.loc[slot_mask, "event_impact_sum"] += impact
    return frame


def _lectures_to_slot_frame(
    *,
    index: pd.DatetimeIndex,
    lecture_df: pd.DataFrame | None,
    slot_minutes: int,
) -> pd.DataFrame:
    if len(index) == 0:
        return _empty_lecture_slot_frame(index)
    if lecture_df is None or lecture_df.empty:
        return _empty_lecture_slot_frame(index)

    df = lecture_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "metadata" not in df.columns:
        df["metadata"] = [{} for _ in range(len(df))]
    for col in ["active_lectures", "starts_next_60m", "ends_next_60m", "quality_score"]:
        if col not in df.columns:
            df[col] = 0.0 if col != "quality_score" else 1.0

    def _meta_float(metadata: Any, key: str, default: float = 0.0) -> float:
        if not isinstance(metadata, dict):
            return default
        try:
            value = float(metadata.get(key, default))
        except Exception:
            return default
        if not math.isfinite(value):
            return default
        return value

    df["lecture_net_pull"] = df["metadata"].apply(lambda metadata: _meta_float(metadata, "lecture_net_pull", 0.0))
    freq = f"{max(1, int(slot_minutes))}min"
    slot = (
        df.set_index("timestamp")[
            ["active_lectures", "starts_next_60m", "ends_next_60m", "lecture_net_pull", "quality_score"]
        ]
        .resample(freq)
        .mean(numeric_only=True)
        .reindex(index)
    )
    slot["active_lectures"] = slot["active_lectures"].interpolate(limit_direction="both").fillna(0.0)
    slot["starts_next_60m"] = slot["starts_next_60m"].interpolate(limit_direction="both").fillna(0.0)
    slot["ends_next_60m"] = slot["ends_next_60m"].interpolate(limit_direction="both").fillna(0.0)
    slot["lecture_net_pull"] = slot["lecture_net_pull"].interpolate(limit_direction="both").fillna(0.0)
    slot["quality_score"] = slot["quality_score"].ffill().bfill().fillna(0.8)
    return slot


def _future_slot_index(
    *,
    last_ts: datetime,
    days: int,
    slot_minutes: int,
) -> pd.DatetimeIndex:
    slot = max(1, int(slot_minutes))
    anchor = pd.Timestamp(_normalize_dt(last_ts)).ceil(f"{slot}min")
    periods = max(1, int(days) * int(round(24 * 60 / slot)))
    return pd.date_range(start=anchor, periods=periods, freq=f"{slot}min", tz="UTC")


def _slot_reference_features(slot_history: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and rolling-mean columns to the slot history frame.

    Adds day_of_week, slot_of_day, is_weekend, and 7-day and 14-day rolling
    mean columns.  These are used to look up the 'typical' pattern for each
    future slot by matching (day_of_week, slot_of_day).
    """
    ref = slot_history.copy()
    ref["day_of_week"] = ref.index.dayofweek
    ref["slot_of_day"] = ref.index.hour * 60 + ref.index.minute
    ref["is_weekend"] = (ref["day_of_week"] >= 5).astype(float)
    ref["rolling_7d"] = ref["occupancy"].rolling(window=min(len(ref), 7 * 24), min_periods=1).mean()
    ref["rolling_14d"] = ref["occupancy"].rolling(window=min(len(ref), 14 * 24), min_periods=1).mean()
    return ref


def _slot_quality_band(score: float) -> str:
    if score >= 0.85:
        return "good"
    if score >= 0.65:
        return "reduced"
    return "poor"


def _build_daily_summaries_from_points(points: list[dict[str, Any]], slot_minutes: int) -> list[dict[str, Any]]:
    if not points:
        return []

    slots_per_day = max(1, int(round((24 * 60) / max(1, int(slot_minutes)))))
    point_frame = pd.DataFrame(points)
    point_frame["timestamp"] = pd.to_datetime(point_frame["timestamp"], utc=True)

    daily_summaries: list[dict[str, Any]] = []
    for day_index in range(0, len(point_frame), slots_per_day):
        day_frame = point_frame.iloc[day_index : day_index + slots_per_day].copy()
        if day_frame.empty:
            continue
        peak_idx = day_frame["yhat"].idxmax()
        peak_row = day_frame.loc[peak_idx]
        avg_yhat = float(day_frame["yhat"].mean())
        day_peak = float(day_frame["yhat"].max())
        if day_peak >= max(1.0, avg_yhat * 1.25):
            risk_level = "high"
        elif day_peak >= max(1.0, avg_yhat * 1.10):
            risk_level = "medium"
        else:
            risk_level = "low"
        first_ts = day_frame["timestamp"].iloc[0]
        daily_summaries.append(
            {
                "date": first_ts.date().isoformat(),
                "peak_slot": peak_row["timestamp"].isoformat(),
                "peak_yhat": float(round(float(peak_row["yhat"]), 4)),
                "avg_yhat": float(round(avg_yhat, 4)),
                "risk_level": risk_level,
                "data_quality": _slot_quality_band(float(day_frame["quality_score"].mean())),
            }
        )
    return daily_summaries


def build_weekly_forecast(
    *,
    zone_id: str,
    history_df: pd.DataFrame,
    events_df: pd.DataFrame | None,
    lecture_df: pd.DataFrame | None,
    days: int = 7,
    slot_minutes: int = 60,
    capacity: float = 100.0,
) -> dict[str, Any]:
    """Build a deterministic 7-day slot-level occupancy forecast.

    For each future slot in the next `days` days, computes:
      - base_mean: historical mean occupancy at this (day_of_week, slot_of_day)
      - event_delta: expected occupancy change from overlapping calendar events
      - lecture_delta: expected occupancy change from lecture transitions
      - yhat = clip(base_mean + event_delta + lecture_delta, 0, capacity)
      - pi_low, pi_high: uncertainty band from historical std

    The result is structured as a list of hourly 'points' and a list of
    daily summaries (peak slot, average, risk level, data quality).

    This forecast is displayed in the dashboard's 'Weekly Outlook' view and
    is also called by the XAI service to generate lecture impact narratives.

    Args:
        zone_id: Zone identifier (included in evidence lineage).
        history_df: Raw occupancy history (last ~28 days recommended).
        events_df: Calendar events with starts_at, ends_at, expected_impact.
        lecture_df: DHBW lecture activity with timestamps and metadata.
        days: Number of days to forecast ahead (default 7).
        slot_minutes: Slot granularity in minutes (default 60 = hourly).
        capacity: Physical room capacity. Predictions are clipped to [0, capacity].

    Returns:
        Dict with keys: zone_id, generated_at, forecast_horizon_days,
        slot_minutes, points (list of per-slot dicts), daily_summaries,
        evidence (lineage metadata), model_info.
    """
    generated_at = datetime.now(UTC)
    slot_history = _history_to_slot_frame(history_df=history_df, slot_minutes=slot_minutes)
    if slot_history.empty:
        future_index = _future_slot_index(last_ts=generated_at, days=days, slot_minutes=slot_minutes)
        points = [
            {
                "timestamp": ts.to_pydatetime().isoformat(),
                "yhat": 0.0,
                "pi_low": 0.0,
                "pi_high": 2.0,
                "day_of_week": int(ts.dayofweek),
                "slot_of_day": int(ts.hour * 60 + ts.minute),
            }
            for ts in future_index
        ]
        daily_summaries = _build_daily_summaries_from_points(points=points, slot_minutes=slot_minutes)
        evidence = {
            "evidence_id": f"weekly-{zone_id}-{generated_at.strftime('%Y%m%d%H%M%S')}",
            "generated_at": generated_at.isoformat(),
            "time_window": {
                "from": (generated_at - timedelta(days=WEEKLY_DEFAULT_HISTORY_DAYS)).isoformat(),
                "to": generated_at.isoformat(),
            },
            "sources": [
                {"type": "counts", "id": f"zone:{zone_id}"},
            ],
            "model": {
                "name": "weekly_reference_pattern",
                "version": "weekly-slot-v1",
                "backend": "deterministic",
                "product": WEEKLY_PRODUCT,
            },
            "quality": {
                "score": 0.0,
                "flags": ["NO_HISTORY"],
            },
        }
        return {
            "zone_id": zone_id,
            "product": WEEKLY_PRODUCT,
            "days": int(days),
            "slot_minutes": int(slot_minutes),
            "generated_at": generated_at.isoformat(),
            "summary": "Weekly slot outlook fallback: no history available.",
            "model_version": "weekly-slot-v1",
            "points": points,
            "daily_summaries": daily_summaries,
            "evidence": evidence,
            "lineage": {
                "product": WEEKLY_PRODUCT,
                "model_run_id": None,
                "backend": "deterministic",
                "model_version": "weekly-slot-v1",
                "scientific_status": "not_trained",
                "feature_set_version": WEEKLY_FEATURE_SET_VERSION,
                "reference_window": evidence["time_window"],
                "reference_objects": [],
            },
        }

    slot_features = _slot_reference_features(slot_history)
    last_ts = _normalize_dt(slot_history.index.max().to_pydatetime())
    future_index = _future_slot_index(last_ts=last_ts, days=days, slot_minutes=slot_minutes)
    future_events = _events_to_slot_frame(index=future_index, events_df=events_df, slot_minutes=slot_minutes)
    future_lectures = _lectures_to_slot_frame(index=future_index, lecture_df=lecture_df, slot_minutes=slot_minutes)

    recent_24h = slot_history.tail(max(1, int(round((24 * 60) / max(1, slot_minutes)))))
    recent_7d = slot_features.tail(max(1, int(round((7 * 24 * 60) / max(1, slot_minutes)))))
    overall_quality = float(slot_history["quality_score"].mean()) if not slot_history.empty else 0.0
    raw_quality_penalty = max(0.0, 1.0 - overall_quality)

    slot_lookup = (
        slot_features.groupby(["day_of_week", "slot_of_day"])
        .agg(
            occupancy_mean=("occupancy", "mean"),
            occupancy_std=("occupancy", lambda s: float(np.std(pd.to_numeric(s, errors="coerce"), ddof=0))),
            utilization_mean=("utilization", "mean"),
            quality_mean=("quality_score", "mean"),
        )
        .reset_index()
    )
    recent_lookup = (
        recent_7d.assign(
            day_of_week=recent_7d.index.dayofweek,
            slot_of_day=recent_7d.index.hour * 60 + recent_7d.index.minute,
        )
        .groupby(["day_of_week", "slot_of_day"])
        .agg(recent_mean=("occupancy", "mean"))
        .reset_index()
    )
    slot_lookup = slot_lookup.merge(recent_lookup, on=["day_of_week", "slot_of_day"], how="left")
    slot_lookup["recent_mean"] = slot_lookup["recent_mean"].fillna(slot_lookup["occupancy_mean"])

    recent_global_mean = float(recent_24h["occupancy"].mean()) if not recent_24h.empty else float(slot_history["occupancy"].mean())
    recent_global_peak = float(recent_24h["occupancy"].max()) if not recent_24h.empty else float(slot_history["occupancy"].max())

    points: list[dict[str, Any]] = []
    for ts in future_index:
        dow = int(ts.dayofweek)
        slot_of_day = int(ts.hour * 60 + ts.minute)
        ref_row = slot_lookup[(slot_lookup["day_of_week"] == dow) & (slot_lookup["slot_of_day"] == slot_of_day)]
        if ref_row.empty:
            base_mean = recent_global_mean
            base_std = max(2.0, float(slot_history["occupancy"].std(ddof=0) or 0.0))
            quality_mean = overall_quality
        else:
            record = ref_row.iloc[0]
            base_mean = float(record.get("occupancy_mean", recent_global_mean) or recent_global_mean)
            recent_mean = float(record.get("recent_mean", base_mean) or base_mean)
            base_mean = 0.65 * base_mean + 0.25 * recent_mean + 0.10 * recent_global_mean
            base_std = max(2.0, float(record.get("occupancy_std", 0.0) or 0.0))
            quality_mean = float(record.get("quality_mean", overall_quality) or overall_quality)

        event_delta = 5.0 * float(future_events.loc[ts, "event_impact_sum"])
        lecture_delta = (
            0.04 * float(future_lectures.loc[ts, "lecture_net_pull"])
            + 1.0 * float(future_lectures.loc[ts, "starts_next_60m"])
            - 0.5 * float(future_lectures.loc[ts, "ends_next_60m"])
        )
        yhat = min(capacity, max(0.0, base_mean + event_delta + lecture_delta))
        uncertainty = max(
            2.0,
            base_std * (1.0 + raw_quality_penalty + max(0.0, 0.5 - quality_mean)),
        )
        pi_low = max(0.0, yhat - uncertainty)
        pi_high = min(capacity, max(pi_low, yhat + uncertainty))
        points.append(
            {
                "timestamp": ts.to_pydatetime().isoformat(),
                "yhat": float(round(yhat, 4)),
                "pi_low": float(round(pi_low, 4)),
                "pi_high": float(round(pi_high, 4)),
                "day_of_week": dow,
                "slot_of_day": slot_of_day,
                "event_active": float(future_events.loc[ts, "event_active"]),
                "event_impact_sum": float(round(future_events.loc[ts, "event_impact_sum"], 4)),
                "lecture_net_pull": float(round(future_lectures.loc[ts, "lecture_net_pull"], 4)),
                "quality_score": float(round(quality_mean, 4)),
            }
        )

    daily_summaries = _build_daily_summaries_from_points(points=points, slot_minutes=slot_minutes)

    reference_window = {
        "from": _normalize_dt(slot_history.index.min().to_pydatetime()).isoformat(),
        "to": last_ts.isoformat(),
    }
    quality_flags: list[str] = []
    if overall_quality < 0.7:
        quality_flags.append("LOW_HISTORY_QUALITY")
    if len(slot_history) < int(round((7 * 24 * 60) / max(1, slot_minutes))):
        quality_flags.append("SHORT_REFERENCE_WINDOW")
    if not quality_flags:
        quality_flags = ["OK"]

    evidence = {
        "evidence_id": f"weekly-{zone_id}-{generated_at.strftime('%Y%m%d%H%M%S')}",
        "generated_at": generated_at.isoformat(),
        "time_window": reference_window,
        "sources": [
            {
                "type": "counts",
                "id": f"zone:{zone_id}",
                "note": f"slot_history={len(slot_history)} slots slot_minutes={slot_minutes}",
            },
            {"type": "events", "id": f"future-events:{len(events_df) if events_df is not None else 0}"},
            {"type": "lecture_activity", "id": f"future-lecture-slots:{len(lecture_df) if lecture_df is not None else 0}"},
        ],
        "model": {
            "name": "weekly_reference_pattern",
            "version": "weekly-slot-v1",
            "backend": "deterministic",
            "product": WEEKLY_PRODUCT,
            "feature_set_version": WEEKLY_FEATURE_SET_VERSION,
        },
        "quality": {
            "score": float(round(max(0.0, min(1.0, overall_quality)), 6)),
            "flags": quality_flags,
        },
    }
    return {
        "zone_id": zone_id,
        "product": WEEKLY_PRODUCT,
        "days": int(days),
        "slot_minutes": int(slot_minutes),
        "generated_at": generated_at.isoformat(),
        "summary": (
            f"Weekly slot outlook based on {len(slot_history)} historical slots, "
            f"{len(points)} forecast slots and internal event/lecture signals."
        ),
        "model_version": "weekly-slot-v1",
        "points": points,
        "daily_summaries": daily_summaries,
        "evidence": evidence,
        "lineage": {
            "product": WEEKLY_PRODUCT,
            "model_run_id": None,
            "backend": "deterministic",
            "model_version": "weekly-slot-v1",
            "scientific_status": "deterministic_reference",
            "feature_set_version": WEEKLY_FEATURE_SET_VERSION,
            "reference_window": reference_window,
            "reference_objects": [],
        },
    }
