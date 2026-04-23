#!/usr/bin/env python3
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "services" / "forecast"))

from scientific_eval import EvaluationConfig, evaluate_training_run  # noqa: E402


def _synthetic_history(days: int = 8) -> pd.DataFrame:
    points = days * 24 * 60
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    start = now - timedelta(minutes=points)
    rows = []
    for i in range(points):
        ts = start + timedelta(minutes=i)
        seasonal = 42 + 10 * np.sin(2 * np.pi * i / 1440.0)
        trend = 0.0005 * i
        occ = max(0.0, seasonal + trend + 1.1 * np.cos(i / 19.0))
        rows.append(
            {
                "timestamp": ts,
                "occupancy": float(occ),
                "utilization": float(min(1.0, occ / 120.0)),
                "quality_score": 0.95,
                "quality_flags": ["OK"],
            }
        )
    return pd.DataFrame(rows)


def _synthetic_lecture(history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ts in pd.to_datetime(history["timestamp"], utc=True):
        minute = int(ts.minute)
        hour = int(ts.hour)
        # Keep lecture_active widely available, heavy_effect intentionally sparse.
        metadata = {
            "heavy_active_lectures": 1 if hour == 2 and minute < 5 else 0,
            "heavy_ended_last_60m": 0,
        }
        rows.append(
            {
                "timestamp": ts,
                "active_lectures": 1,
                "active_courses": 1,
                "starts_next_60m": 1 if minute == 0 else 0,
                "ends_next_60m": 1 if minute == 30 else 0,
                "quality_score": 1.0,
                "quality_flags": ["LECTURE_RAPLA"],
                "metadata": metadata,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    history = _synthetic_history(days=8)
    events = pd.DataFrame(columns=["starts_at", "ends_at", "expected_impact"])
    lecture = _synthetic_lecture(history)

    config = EvaluationConfig(
        zone_id="default-zone",
        horizons=[60],
        folds=2,
        train_days=1,
        val_days=1,
        test_days=1,
        gap_minutes=0,
        origin_stride_minutes=60,
        max_origins_per_split=200,
        enable_tf=False,
        enable_sarimax=False,
        primary_horizon=60,
        segment_min_samples=30,
    )

    report = evaluate_training_run(
        history_df=history,
        events_df=events,
        lecture_df=lecture,
        config=config,
    )

    baseline_h60 = (
        report.get("models", {})
        .get("baseline", {})
        .get("horizons", {})
        .get("60", {})
    )
    segments = baseline_h60.get("segments", {})
    if not isinstance(segments, dict):
        raise AssertionError("segments payload missing in baseline horizon report")

    required = {
        "lecture_active",
        "heavy_effect",
        "lecture_transition_start",
        "lecture_transition_end",
    }
    if not required.issubset(set(segments.keys())):
        raise AssertionError(f"missing segment keys: {required - set(segments.keys())}")

    lecture_active = segments["lecture_active"]
    heavy_effect = segments["heavy_effect"]

    if bool(lecture_active.get("test_insufficient_samples", True)):
        raise AssertionError("lecture_active should have enough samples")

    if not bool(heavy_effect.get("test_insufficient_samples", False)):
        raise AssertionError("heavy_effect should be marked as insufficient for sparse synthetic setup")

    heavy_metrics = heavy_effect.get("test_metrics", {})
    if heavy_metrics.get("mae") is not None:
        raise AssertionError("insufficient heavy_effect segment must not expose evaluated MAE")

    print("forecast scientific segment tests passed")


if __name__ == "__main__":
    main()
