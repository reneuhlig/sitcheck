"""Dry Run: Technical validation of the GBDT forecast pipeline.

Tests reproducibility, feature alignment, model roundtrip,
quantile consistency, and inference format correctness.

Usage:
    python3 scripts/evaluation/dry_run.py \
        --data training_data.parquet \
        --model-dir services/forecast/models
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import hashlib
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import joblib

# Add forecast service to path
FORECAST_DIR = Path(__file__).resolve().parent.parent.parent / "services" / "forecast"
sys.path.insert(0, str(FORECAST_DIR))

from features_excel import (
    FEATURE_COLUMNS,
    FEATURE_SET_VERSION,
    CAPACITY_TOTAL,
    build_supervised_dataset,
    load_training_data,
    walk_forward_splits,
)
from model_gbdt import GBDTBundle, GBDTConfig, predict_gbdt, train_gbdt_quantiles
from model_store import bundle_dir


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    details: dict = field(default_factory=dict)


class DryRunSuite:
    """Collects and reports test results."""

    def __init__(self) -> None:
        self.results: list[TestResult] = []

    def add(self, name: str, passed: bool, message: str, **details: object) -> None:
        self.results.append(TestResult(name, passed, message, details))
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {message}")

    def summary(self) -> dict:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "all_passed": failed == 0,
            "results": [
                {"name": r.name, "passed": r.passed, "message": r.message, **r.details}
                for r in self.results
            ],
        }


# ---------------------------------------------------------------------------
# D1: Data loading reproducibility
# ---------------------------------------------------------------------------

def test_data_loading(suite: DryRunSuite, parquet_path: str) -> pd.DataFrame:
    """D1: Verify data loads deterministically."""
    print("\n=== D1: Data Loading Reproducibility ===")
    df1 = load_training_data(parquet_path)
    df2 = load_training_data(parquet_path)

    same_shape = df1.shape == df2.shape
    same_values = df1.equals(df2)

    suite.add(
        "D1.1 Shape consistency",
        same_shape,
        f"Shape: {df1.shape}",
    )
    suite.add(
        "D1.2 Value determinism",
        same_values,
        "Two loads produce identical DataFrames" if same_values else "DataFrames differ!",
    )

    # Check expected columns
    required = ["timestamp", "occupancy"]
    missing = [c for c in required if c not in df1.columns]
    suite.add(
        "D1.3 Required columns present",
        len(missing) == 0,
        f"Missing: {missing}" if missing else "timestamp, occupancy present",
    )

    # Check timestamp ordering
    ts_sorted = df1["timestamp"].is_monotonic_increasing
    suite.add(
        "D1.4 Timestamps monotonically increasing",
        ts_sorted,
        "Chronological order confirmed" if ts_sorted else "Timestamps NOT sorted!",
    )

    return df1


# ---------------------------------------------------------------------------
# D2: Feature engineering determinism
# ---------------------------------------------------------------------------

def test_feature_engineering(suite: DryRunSuite, df: pd.DataFrame) -> object:
    """D2: Verify feature engineering is deterministic."""
    print("\n=== D2: Feature Engineering Determinism ===")
    ds1 = build_supervised_dataset(df, horizon_steps=4)
    ds2 = build_supervised_dataset(df, horizon_steps=4)

    same_X = np.array_equal(ds1.X.values, ds2.X.values)
    same_y = np.array_equal(ds1.y, ds2.y)
    same_cols = ds1.feature_columns == ds2.feature_columns

    suite.add("D2.1 X deterministic", same_X, "Feature matrices identical")
    suite.add("D2.2 y deterministic", same_y, "Target vectors identical")
    suite.add("D2.3 Feature columns deterministic", same_cols, f"{len(ds1.feature_columns)} features")

    # Check no NaNs in dataset
    nan_count = int(ds1.X.isna().sum().sum())
    suite.add("D2.4 No NaN in features", nan_count == 0, f"NaN count: {nan_count}")

    nan_y = int(np.isnan(ds1.y).sum())
    suite.add("D2.5 No NaN in target", nan_y == 0, f"NaN count: {nan_y}")

    # Feature column coverage
    available = [c for c in FEATURE_COLUMNS if c in ds1.feature_columns]
    suite.add(
        "D2.6 Feature coverage",
        len(available) == len(FEATURE_COLUMNS),
        f"{len(available)}/{len(FEATURE_COLUMNS)} features available",
    )

    return ds1


# ---------------------------------------------------------------------------
# D3: Walk-forward splits validity
# ---------------------------------------------------------------------------

def test_walk_forward_splits(suite: DryRunSuite, dataset: object) -> list:
    """D3: Verify walk-forward splits are chronologically valid."""
    print("\n=== D3: Walk-Forward Splits ===")
    folds = walk_forward_splits(dataset, n_folds=6)

    suite.add("D3.1 Folds generated", len(folds) > 0, f"{len(folds)} folds")

    all_non_overlapping = True
    all_chronological = True

    for fold in folds:
        train_idx = fold["train_idx"]
        val_idx = fold["val_idx"]
        test_idx = fold["test_idx"]

        # No overlap between sets
        train_set = set(train_idx)
        val_set = set(val_idx)
        test_set = set(test_idx)

        if train_set & val_set or train_set & test_set or val_set & test_set:
            all_non_overlapping = False

        # Chronological: max(train) < min(val) < min(test)
        ts = dataset.timestamps
        train_ts_max = ts.iloc[train_idx[-1]]
        val_ts_min = ts.iloc[val_idx[0]]
        test_ts_min = ts.iloc[test_idx[0]]

        if not (train_ts_max < val_ts_min <= test_ts_min):
            all_chronological = False

    suite.add(
        "D3.2 No set overlap",
        all_non_overlapping,
        "Train/val/test sets are disjoint" if all_non_overlapping else "OVERLAP DETECTED!",
    )
    suite.add(
        "D3.3 Chronological order",
        all_chronological,
        "train < val < test in all folds" if all_chronological else "ORDER VIOLATION!",
    )

    # Expanding window: each fold's training set should be >= previous
    expanding = all(
        folds[i]["train_size"] <= folds[i + 1]["train_size"]
        for i in range(len(folds) - 1)
    )
    suite.add(
        "D3.4 Expanding window",
        expanding,
        "Training sets grow monotonically" if expanding else "NOT expanding!",
    )

    return folds


# ---------------------------------------------------------------------------
# D4: Model load-predict roundtrip
# ---------------------------------------------------------------------------

def test_model_roundtrip(suite: DryRunSuite, model_dir: str, zone_id: str, horizon: int, dataset: object) -> None:
    """D4: Verify saved model produces consistent predictions."""
    print("\n=== D4: Model Load-Predict Roundtrip ===")
    root = bundle_dir(model_dir, zone_id, horizon)

    # Check files exist
    expected_files = ["gbdt_q10.joblib", "gbdt_q50.joblib", "gbdt_q90.joblib", "metadata_gbdt.json", "feature_importance.json"]
    missing = [f for f in expected_files if not (root / f).exists()]
    suite.add(
        "D4.1 Model files present",
        len(missing) == 0,
        f"All {len(expected_files)} files found" if not missing else f"Missing: {missing}",
    )

    if missing:
        return

    # Load models
    model_q10 = joblib.load(root / "gbdt_q10.joblib")
    model_q50 = joblib.load(root / "gbdt_q50.joblib")
    model_q90 = joblib.load(root / "gbdt_q90.joblib")

    metadata = json.loads((root / "metadata_gbdt.json").read_text())
    fi = json.loads((root / "feature_importance.json").read_text())

    bundle = GBDTBundle(
        model_q10=model_q10,
        model_q50=model_q50,
        model_q90=model_q90,
        feature_columns=metadata.get("feature_columns", []),
        feature_importance=fi,
        config=GBDTConfig(),
    )

    # Predict on a small sample
    sample_idx = list(range(min(100, len(dataset.X))))
    X_sample = dataset.X.iloc[sample_idx].values.astype(np.float32)

    preds1 = predict_gbdt(bundle, X_sample, capacity=CAPACITY_TOTAL)
    preds2 = predict_gbdt(bundle, X_sample, capacity=CAPACITY_TOTAL)

    deterministic = (
        np.array_equal(preds1["q03"], preds2["q03"])
        and np.array_equal(preds1["q50"], preds2["q50"])
        and np.array_equal(preds1["q97"], preds2["q97"])
    )
    suite.add(
        "D4.2 Prediction determinism",
        deterministic,
        "Two predict calls produce identical outputs",
    )

    # Metadata checks
    suite.add(
        "D4.3 Model promoted",
        metadata.get("promoted", False),
        f"promoted={metadata.get('promoted')}",
    )

    stored_cols = metadata.get("feature_columns", [])
    suite.add(
        "D4.4 Feature columns in metadata",
        len(stored_cols) == len(dataset.feature_columns),
        f"{len(stored_cols)} stored vs {len(dataset.feature_columns)} expected",
    )

    # Quantile levels
    ql = metadata.get("quantile_levels", {})
    correct_levels = ql.get("lower") == 0.03 and ql.get("upper") == 0.97
    suite.add(
        "D4.5 Quantile levels correct (0.03/0.97)",
        correct_levels,
        f"lower={ql.get('lower')}, upper={ql.get('upper')}",
    )


# ---------------------------------------------------------------------------
# D5: Quantile consistency
# ---------------------------------------------------------------------------

def test_quantile_consistency(suite: DryRunSuite, model_dir: str, zone_id: str, horizon: int, dataset: object) -> None:
    """D5: Verify q03 <= q50 <= q97 for all predictions."""
    print("\n=== D5: Quantile Consistency ===")
    root = bundle_dir(model_dir, zone_id, horizon)

    model_q10 = joblib.load(root / "gbdt_q10.joblib")
    model_q50 = joblib.load(root / "gbdt_q50.joblib")
    model_q90 = joblib.load(root / "gbdt_q90.joblib")
    fi = json.loads((root / "feature_importance.json").read_text())
    metadata = json.loads((root / "metadata_gbdt.json").read_text())

    bundle = GBDTBundle(
        model_q10=model_q10, model_q50=model_q50, model_q90=model_q90,
        feature_columns=metadata.get("feature_columns", []),
        feature_importance=fi, config=GBDTConfig(),
    )

    # Predict on ALL test data
    X_all = dataset.X.values.astype(np.float32)
    preds = predict_gbdt(bundle, X_all, capacity=CAPACITY_TOTAL)

    q03 = preds["q03"]
    q50 = preds["q50"]
    q97 = preds["q97"]

    # Monotonicity: q03 <= q50 <= q97
    mono_lower = np.all(q03 <= q50 + 1e-6)
    mono_upper = np.all(q50 <= q97 + 1e-6)

    suite.add("D5.1 q03 <= q50", mono_lower, f"Violations: {np.sum(q03 > q50 + 1e-6)}")
    suite.add("D5.2 q50 <= q97", mono_upper, f"Violations: {np.sum(q50 > q97 + 1e-6)}")

    # Range check: all predictions in [0, CAPACITY_TOTAL]
    in_range = np.all(q03 >= 0) and np.all(q97 <= CAPACITY_TOTAL)
    suite.add(
        "D5.3 Predictions in [0, 84]",
        in_range,
        f"q03 range: [{q03.min():.1f}, {q03.max():.1f}], q97 range: [{q97.min():.1f}, {q97.max():.1f}]",
    )

    # Spread statistics
    spread = q97 - q03
    suite.add(
        "D5.4 Spread statistics",
        True,
        f"Mean spread: {spread.mean():.2f}, Max: {spread.max():.2f}, Min: {spread.min():.2f}",
    )

    # Non-negative spread
    non_neg_spread = np.all(spread >= -1e-6)
    suite.add("D5.5 Non-negative spread", non_neg_spread, f"Negative spreads: {np.sum(spread < -1e-6)}")


# ---------------------------------------------------------------------------
# D6: Feature alignment training <-> inference
# ---------------------------------------------------------------------------

def test_feature_alignment(suite: DryRunSuite, model_dir: str, zone_id: str, horizon: int) -> None:
    """D6: Verify training features match what inference expects."""
    print("\n=== D6: Feature Alignment Training <-> Inference ===")
    root = bundle_dir(model_dir, zone_id, horizon)
    metadata = json.loads((root / "metadata_gbdt.json").read_text())

    stored_cols = metadata.get("feature_columns", [])
    expected_cols = list(FEATURE_COLUMNS)

    # Check stored == expected
    match = stored_cols == [c for c in expected_cols if c in stored_cols]
    suite.add(
        "D6.1 Feature order matches FEATURE_COLUMNS",
        stored_cols == expected_cols,
        f"Stored: {len(stored_cols)}, Expected: {len(expected_cols)}",
    )

    # Check feature_set_version
    fsv = metadata.get("feature_set_version", "")
    suite.add(
        "D6.2 Feature set version",
        fsv == FEATURE_SET_VERSION,
        f"Stored: '{fsv}', Expected: '{FEATURE_SET_VERSION}'",
    )

    # Check LIVE_TO_EXCEL mapping coverage
    # These are the features that CAN be mapped from live inference
    LIVE_MAPPED_FEATURES = {
        "occupancy_lag_1", "occupancy_lag_2", "occupancy_lag_3", "occupancy_lag_4",
        "occupancy_lag_8", "occupancy_lag_16",
        "occupancy_roll_mean_4", "occupancy_roll_mean_8", "occupancy_roll_mean_16",
        "occupancy_roll_std_4", "occupancy_roll_std_8",
        "occupancy_diff_1", "occupancy_diff_4",
        "minute_sin", "minute_cos", "dow_sin", "dow_cos",
        "day_of_year_sin", "day_of_year_cos", "hour_of_day", "is_weekday",
        "lecture_density_proxy", "lecture_starts_proxy", "lecture_ends_proxy", "lecture_heavy_proxy",
        "utilization_pct",
        # Derived from timestamp during inference
        "f_month", "f_weekday", "f_tod", "f_weather", "f_bridge",
        "efficiency", "capacity_effective",
    }
    # Features that will be 0-filled during live inference (no mapping)
    unmapped = [c for c in stored_cols if c not in LIVE_MAPPED_FEATURES]
    suite.add(
        "D6.3 Unmapped features (0-filled in inference)",
        True,  # informational
        f"{len(unmapped)} features unmapped: {unmapped}",
        unmapped_features=unmapped,
    )

    # Critical: check if any top-importance features are unmapped
    fi = json.loads((root / "feature_importance.json").read_text())
    top10 = list(fi.keys())[:10]
    unmapped_top10 = [f for f in top10 if f in unmapped]
    suite.add(
        "D6.4 Top-10 importance features mapped",
        len(unmapped_top10) == 0,
        f"Unmapped in top-10: {unmapped_top10}" if unmapped_top10 else "All top-10 features have live mapping",
    )


# ---------------------------------------------------------------------------
# D7: Feature labels completeness
# ---------------------------------------------------------------------------

def test_feature_labels(suite: DryRunSuite, model_dir: str, zone_id: str, horizon: int) -> None:
    """D7: Verify all features have German labels."""
    print("\n=== D7: Feature Labels ===")

    xai_dir = FORECAST_DIR.parent / "xai"
    sys.path.insert(0, str(xai_dir))

    try:
        from feature_labels import FEATURE_LABELS_DE as FEATURE_LABELS
    except ImportError as e:
        suite.add("D7.1 feature_labels.py importable", False, f"Import failed: {e}")
        return

    suite.add("D7.1 feature_labels.py importable", True, f"{len(FEATURE_LABELS)} labels defined")

    root = bundle_dir(model_dir, zone_id, horizon)
    metadata = json.loads((root / "metadata_gbdt.json").read_text())
    stored_cols = metadata.get("feature_columns", [])

    missing_labels = [c for c in stored_cols if c not in FEATURE_LABELS]
    suite.add(
        "D7.2 All training features have labels",
        len(missing_labels) == 0,
        f"Missing: {missing_labels}" if missing_labels else f"All {len(stored_cols)} features labeled",
    )


# ---------------------------------------------------------------------------
# D8: Error handling
# ---------------------------------------------------------------------------

def test_error_handling(suite: DryRunSuite, parquet_path: str) -> None:
    """D8: Verify graceful error handling for edge cases."""
    print("\n=== D8: Error Handling ===")

    # Empty DataFrame
    try:
        build_supervised_dataset(pd.DataFrame(), horizon_steps=4)
        suite.add("D8.1 Empty DataFrame raises", False, "No error raised!")
    except (ValueError, KeyError):
        suite.add("D8.1 Empty DataFrame raises ValueError", True, "Correct exception raised")

    # Invalid horizon
    df = load_training_data(parquet_path)
    try:
        build_supervised_dataset(df, horizon_steps=0)
        suite.add("D8.2 horizon_steps=0 raises", False, "No error raised!")
    except ValueError:
        suite.add("D8.2 horizon_steps=0 raises ValueError", True, "Correct exception raised")

    # Missing model directory
    try:
        from train_gbdt import load_gbdt_bundle
        load_gbdt_bundle("/nonexistent", "x", 60)
        suite.add("D8.3 Missing model raises", False, "No error raised!")
    except (FileNotFoundError, Exception):
        suite.add("D8.3 Missing model raises error", True, "Correct exception raised")


# ---------------------------------------------------------------------------
# D9: CV metrics sanity
# ---------------------------------------------------------------------------

def test_metrics_sanity(suite: DryRunSuite, model_dir: str, zone_id: str, horizon: int) -> None:
    """D9: Verify stored metrics are plausible."""
    print("\n=== D9: Metrics Sanity ===")
    root = bundle_dir(model_dir, zone_id, horizon)
    metadata = json.loads((root / "metadata_gbdt.json").read_text())
    metrics = metadata.get("metrics", {})

    mae = metrics.get("cv_weighted_mae", 999)
    suite.add("D9.1 MAE > 0", mae > 0, f"MAE = {mae:.3f}")
    suite.add("D9.2 MAE < 10 (plausible)", mae < 10, f"MAE = {mae:.3f}")

    cov = metrics.get("cv_weighted_coverage_94", 0)
    suite.add("D9.3 Coverage in [50, 100]", 50 <= cov <= 100, f"Coverage = {cov:.1f}%")

    bl_pers = metrics.get("cv_weighted_mae_bl_persistence_h60", 0)
    suite.add("D9.4 Baseline MAE > GBDT MAE", bl_pers > mae, f"Baseline={bl_pers:.3f} > GBDT={mae:.3f}")

    improvement = metrics.get("cv_improvement_vs_persistence_pct", 0)
    suite.add("D9.5 Improvement > 0%", improvement > 0, f"Improvement = {improvement:.1f}%")

    # Check fold results stored
    fold_results = metadata.get("fold_results", [])
    suite.add("D9.6 Fold results stored", len(fold_results) > 0, f"{len(fold_results)} folds stored")

    # All folds have required keys
    required_keys = {"mae_gbdt", "mae_bl_persistence_h60", "coverage_94", "test_size"}
    if fold_results:
        fold_keys = set(fold_results[0].keys())
        has_keys = required_keys.issubset(fold_keys)
        suite.add("D9.7 Fold results have required keys", has_keys,
                   f"Missing: {required_keys - fold_keys}" if not has_keys else "All keys present")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Dry Run: GBDT pipeline validation")
    parser.add_argument("--data", required=True, help="Path to training Parquet")
    parser.add_argument("--model-dir", default="models", help="Model directory")
    parser.add_argument("--zone-id", default="default-zone")
    parser.add_argument("--horizon", type=int, default=60, help="Horizon in minutes")
    args = parser.parse_args()

    print("=" * 60)
    print("GBDT FORECAST PIPELINE -- DRY RUN")
    print("=" * 60)

    suite = DryRunSuite()

    # D1: Data loading
    df = test_data_loading(suite, args.data)

    # D2: Feature engineering
    dataset = test_feature_engineering(suite, df)

    # D3: Walk-forward splits
    test_walk_forward_splits(suite, dataset)

    # D4: Model roundtrip
    test_model_roundtrip(suite, args.model_dir, args.zone_id, args.horizon, dataset)

    # D5: Quantile consistency
    test_quantile_consistency(suite, args.model_dir, args.zone_id, args.horizon, dataset)

    # D6: Feature alignment
    test_feature_alignment(suite, args.model_dir, args.zone_id, args.horizon)

    # D7: Feature labels
    test_feature_labels(suite, args.model_dir, args.zone_id, args.horizon)

    # D8: Error handling
    test_error_handling(suite, args.data)

    # D9: Metrics sanity
    test_metrics_sanity(suite, args.model_dir, args.zone_id, args.horizon)

    # Summary
    summary = suite.summary()
    print("\n" + "=" * 60)
    print(f"DRY RUN SUMMARY: {summary['passed']}/{summary['total']} passed, "
          f"{summary['failed']} failed")
    print("=" * 60)

    if summary["all_passed"]:
        print("STATUS: ALL TESTS PASSED")
    else:
        print("STATUS: FAILURES DETECTED")
        for r in summary["results"]:
            if not r["passed"]:
                print(f"  FAIL: {r['name']} -- {r['message']}")

    # Save report
    report_path = Path(args.model_dir) / "dry_run_report.json"
    report_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nReport saved to: {report_path}")

    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
