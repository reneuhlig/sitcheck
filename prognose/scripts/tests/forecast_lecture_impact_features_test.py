#!/usr/bin/env python3
from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FORECAST_DIR = ROOT / "services" / "forecast"
if str(FORECAST_DIR) not in sys.path:
    sys.path.insert(0, str(FORECAST_DIR))

from features import build_feature_frame  # noqa: E402


def _build_history_frame() -> pd.DataFrame:
    idx = pd.date_range(start=datetime(2026, 1, 8, 8, 0, tzinfo=UTC), periods=220, freq="1min")
    return pd.DataFrame(
        {
            "timestamp": idx,
            "occupancy": [float(30 + (i % 20)) for i in range(len(idx))],
            "utilization": [float((30 + (i % 20)) / 100.0) for i in range(len(idx))],
            "quality_score": [0.95 for _ in range(len(idx))],
            "quality_flags": [[] for _ in range(len(idx))],
        }
    )


def _build_lecture_frame(with_metadata: bool) -> pd.DataFrame:
    idx = pd.date_range(start=datetime(2026, 1, 8, 8, 0, tzinfo=UTC), periods=220, freq="1min")
    records = []
    for ts in idx:
        active = 2 if ts.hour in {8, 9} else 1
        heavy_now = 1 if ts.hour == 9 else 0
        heavy_post = 1 if ts.hour == 10 else 0
        pull = 20.0 * active
        bonus = 4.0 * (heavy_now + heavy_post)
        record = {
            "timestamp": ts,
            "active_lectures": active,
            "active_courses": max(1, active - 1),
            "starts_next_60m": 3,
            "ends_next_60m": 2,
            "quality_score": 0.9,
            "quality_flags": ["LECTURE_RAPLA"],
        }
        if with_metadata:
            record["metadata"] = {
                "heavy_active_lectures": heavy_now,
                "heavy_ended_last_60m": heavy_post,
                "lecture_pull_regular": pull,
                "heavy_bib_bonus": bonus,
                "lecture_net_pull": pull - bonus,
                "impact_model_version": "lecture-impact-v1",
            }
        records.append(record)
    return pd.DataFrame(records)


def main() -> int:
    history_df = _build_history_frame()
    lecture_with_meta = _build_lecture_frame(with_metadata=True)
    lecture_without_meta = _build_lecture_frame(with_metadata=False)

    frame_with_meta = build_feature_frame(
        history_df=history_df,
        events_df=pd.DataFrame(),
        lecture_df=lecture_with_meta,
        use_calendar_features=True,
    )
    if frame_with_meta.empty:
        raise AssertionError("feature frame with metadata is empty")

    required_cols = [
        "lecture_heavy_now",
        "lecture_heavy_post_60m",
        "lecture_pull_regular",
        "lecture_bib_bonus",
        "lecture_net_pull",
    ]
    for col in required_cols:
        if col not in frame_with_meta.columns:
            raise AssertionError(f"missing expected feature column: {col}")

    if frame_with_meta["lecture_heavy_now"].max() <= 0:
        raise AssertionError("lecture_heavy_now should include positive values with metadata")
    if frame_with_meta["lecture_net_pull"].max() <= 0:
        raise AssertionError("lecture_net_pull should include positive values with metadata")

    frame_without_meta = build_feature_frame(
        history_df=history_df,
        events_df=pd.DataFrame(),
        lecture_df=lecture_without_meta,
        use_calendar_features=True,
    )
    if frame_without_meta.empty:
        raise AssertionError("feature frame without metadata is empty")

    if frame_without_meta["lecture_heavy_now"].max() != 0.0:
        raise AssertionError("lecture_heavy_now should fallback to 0 without metadata")
    if frame_without_meta["lecture_heavy_post_60m"].max() != 0.0:
        raise AssertionError("lecture_heavy_post_60m should fallback to 0 without metadata")
    if frame_without_meta["lecture_pull_regular"].max() <= 0.0:
        raise AssertionError("lecture_pull_regular should fallback from active_lectures")

    frame_ablation = build_feature_frame(
        history_df=history_df,
        events_df=pd.DataFrame(),
        lecture_df=lecture_with_meta,
        use_calendar_features=True,
        include_lecture_impact=False,
    )
    if frame_ablation.empty:
        raise AssertionError("feature frame for lecture ablation is empty")
    lecture_cols = [
        "lecture_count_now",
        "lecture_starts_next_60m",
        "lecture_ends_next_60m",
        "lecture_pull_regular",
        "lecture_net_pull",
    ]
    for col in lecture_cols:
        if frame_ablation[col].abs().max() != 0.0:
            raise AssertionError(f"{col} should be zeroed when include_lecture_impact is false")

    print("forecast lecture impact feature tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
