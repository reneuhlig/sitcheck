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

from train_tf import TrainingConfig, train_zone_model  # noqa: E402


def _tf_available() -> bool:
    try:
        import tensorflow  # noqa: F401

        return True
    except Exception:
        return False


def main() -> None:
    if not _tf_available():
        print("forecast tf train smoke: tensorflow unavailable, skipping")
        return

    points = 1300
    start = datetime.now(UTC) - timedelta(minutes=points)
    rows = []
    for i in range(points):
        ts = start + timedelta(minutes=i)
        occ = 38 + 10 * np.sin(i / 18.0) + 3 * np.cos(i / 47.0)
        rows.append(
            {
                "timestamp": ts,
                "occupancy": float(max(0.0, occ)),
                "utilization": float(max(0.0, occ / 100.0)),
                "quality_score": 0.94,
                "quality_flags": ["OK"],
            }
        )
    history_df = pd.DataFrame(rows)
    events_df = pd.DataFrame(
        [
            {
                "starts_at": datetime.now(UTC) - timedelta(hours=4),
                "ends_at": datetime.now(UTC) - timedelta(hours=2),
                "expected_impact": 0.2,
            }
        ]
    )

    with tempfile.TemporaryDirectory() as tmp:
        cfg = TrainingConfig(
            model_dir=tmp,
            zone_id="default-zone",
            horizon=12,
            min_train_points=200,
            use_calendar_features=True,
            epochs=3,
            batch_size=64,
            verbose=0,
        )
        result = train_zone_model(history_df=history_df, events_df=events_df, lecture_df=None, config=cfg)
        if result.get("status") != "ok":
            raise AssertionError("expected ok status")
        metrics = result.get("metrics", {})
        required = {"mae_val", "smape_val", "baseline_mae_val", "improvement_vs_baseline_mae_val"}
        if not required.issubset(metrics.keys()):
            raise AssertionError(f"missing metrics keys: {required - set(metrics.keys())}")

    print("forecast tf train smoke passed")


if __name__ == "__main__":
    main()
