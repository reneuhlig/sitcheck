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

from features import build_feature_frame, build_supervised_dataset  # noqa: E402
from model_store import load_bundle  # noqa: E402
from train_tf import TrainingConfig, train_zone_model  # noqa: E402


def _tf_available() -> bool:
    try:
        import tensorflow  # noqa: F401

        return True
    except Exception:
        return False


def _synthetic_history(points: int = 900) -> pd.DataFrame:
    start = datetime.now(UTC) - timedelta(minutes=points)
    rows = []
    for i in range(points):
        ts = start + timedelta(minutes=i)
        occ = 40 + 12 * np.sin(i / 15.0) + 5 * np.cos(i / 37.0)
        util = max(0.0, min(1.0, occ / 100.0))
        rows.append(
            {
                "timestamp": ts,
                "occupancy": float(max(0.0, occ)),
                "utilization": util,
                "quality_score": 0.95,
                "quality_flags": ["OK"],
            }
        )
    return pd.DataFrame(rows)


def _synthetic_events() -> pd.DataFrame:
    now = datetime.now(UTC)
    return pd.DataFrame(
        [
            {
                "starts_at": now - timedelta(hours=2),
                "ends_at": now + timedelta(hours=1),
                "expected_impact": 0.25,
            }
        ]
    )


def test_feature_build_deterministic() -> None:
    history = _synthetic_history(points=300)
    events = _synthetic_events()
    a = build_feature_frame(history_df=history, events_df=events, use_calendar_features=True)
    b = build_feature_frame(history_df=history, events_df=events, use_calendar_features=True)
    if not a.equals(b):
        raise AssertionError("feature frame should be deterministic")


def test_target_alignment() -> None:
    ts0 = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for i in range(200):
        rows.append(
            {
                "timestamp": ts0 + timedelta(minutes=i),
                "occupancy": float(i),
                "utilization": 0.5,
                "quality_score": 1.0,
                "quality_flags": ["OK"],
            }
        )
    history = pd.DataFrame(rows)
    feature_df = build_feature_frame(history_df=history, events_df=pd.DataFrame(), use_calendar_features=False)
    dataset = build_supervised_dataset(feature_df, horizon=3)
    first_idx = dataset.X.index[0]
    base_occ = float(feature_df.loc[first_idx, "occupancy"])
    y_row = dataset.y.loc[first_idx]
    if abs(float(y_row["y_tplus_1"]) - (base_occ + 1)) > 1e-6:
        raise AssertionError("y_tplus_1 alignment mismatch")
    if abs(float(y_row["y_tplus_3"]) - (base_occ + 3)) > 1e-6:
        raise AssertionError("y_tplus_3 alignment mismatch")


def test_model_store_roundtrip() -> None:
    if not _tf_available():
        print("forecast tf unit: tensorflow unavailable in current interpreter, skipping model roundtrip")
        return

    history = _synthetic_history(points=1200)
    events = _synthetic_events()

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
        result = train_zone_model(history_df=history, events_df=events, lecture_df=None, config=cfg)
        if result.get("status") != "ok":
            raise AssertionError("training did not return ok status")

        bundle = load_bundle(tmp, "default-zone", 12)
        sample = history.tail(180).copy()
        feature_df = build_feature_frame(sample, events, use_calendar_features=True)
        vector = feature_df[bundle["metadata"]["feature_columns"]].dropna().tail(1)
        X = bundle["scaler"].transform(vector.values.astype(np.float32)).astype(np.float32)
        pred = bundle["model"].predict(X, verbose=0)
        if pred.shape != (1, 12):
            raise AssertionError(f"unexpected prediction shape: {pred.shape}")


def main() -> None:
    test_feature_build_deterministic()
    test_target_alignment()
    test_model_store_roundtrip()
    print("forecast tf unit tests passed")


if __name__ == "__main__":
    main()
