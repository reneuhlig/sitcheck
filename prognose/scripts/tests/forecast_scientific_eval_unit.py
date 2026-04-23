#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "services" / "forecast"))

from scientific_eval import (  # noqa: E402
    EvaluationConfig,
    evaluate_training_run,
    load_evaluation_report,
    load_latest_evaluation_report,
    save_evaluation_report,
)


def _synthetic_history(days: int = 6) -> pd.DataFrame:
    points = days * 24 * 60
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    start = now - timedelta(minutes=points)
    rows = []
    for i in range(points):
        ts = start + timedelta(minutes=i)
        seasonal = 45 + 12 * np.sin(2 * np.pi * i / 1440.0)
        trend = 0.0008 * i
        noise = 1.5 * np.sin(i / 17.0)
        occ = max(0.0, seasonal + trend + noise)
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


def main() -> None:
    history = _synthetic_history(days=6)
    events = pd.DataFrame(columns=["starts_at", "ends_at", "expected_impact"])
    lecture = pd.DataFrame(
        columns=[
            "timestamp",
            "active_lectures",
            "active_courses",
            "starts_next_60m",
            "ends_next_60m",
            "quality_score",
            "quality_flags",
        ]
    )

    config = EvaluationConfig(
        zone_id="default-zone",
        horizons=[60],
        folds=2,
        train_days=1,
        val_days=1,
        test_days=1,
        gap_minutes=0,
        origin_stride_minutes=180,
        max_origins_per_split=4,
        enable_tf=False,
        enable_sarimax=False,
        primary_horizon=60,
    )

    report = evaluate_training_run(
        history_df=history,
        events_df=events,
        lecture_df=lecture,
        config=config,
    )

    if report.get("zone_id") != "default-zone":
        raise AssertionError("zone_id mismatch")
    if not report.get("run_id", "").startswith("eval-"):
        raise AssertionError("run_id missing")

    models = report.get("models", {})
    baseline = models.get("baseline", {}).get("horizons", {}).get("60", {})
    gbdt = models.get("quantile_gbdt", {}).get("horizons", {}).get("60", {})

    if baseline.get("test_metrics", {}).get("n", 0) <= 0:
        raise AssertionError("baseline test metrics missing")
    if gbdt.get("test_metrics", {}).get("n", 0) <= 0:
        raise AssertionError("gbdt test metrics missing")
    if not isinstance(baseline.get("segments"), dict):
        raise AssertionError("baseline segment metrics missing")

    with tempfile.TemporaryDirectory() as tmp:
        store = save_evaluation_report(model_dir=tmp, zone_id="default-zone", report=report)
        run = load_evaluation_report(model_dir=tmp, zone_id="default-zone", run_id=store["run_id"])
        latest = load_latest_evaluation_report(model_dir=tmp, zone_id="default-zone")
        if run is None or latest is None:
            raise AssertionError("report persistence failed")

    print("forecast scientific eval unit tests passed")


if __name__ == "__main__":
    main()
