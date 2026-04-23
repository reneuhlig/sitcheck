"""GBDT training pipeline for occupancy forecasting.

This module orchestrates the complete training workflow for the primary
LightGBM quantile regression model.  It connects all other components:

  prepare_training_data.py  →  (training_data.parquet)
         ↓
  features_excel.py         →  build_supervised_dataset, walk_forward_splits
         ↓
  model_gbdt.py             →  train_gbdt_quantiles, predict_gbdt
         ↓
  model_store.py            →  save_gbdt_bundle

Pipeline overview (run_full_pipeline):
  1. Load the Parquet training data.
  2. Build a supervised dataset (feature matrix X, target vector y).
  3. Run 6-fold walk-forward cross-validation to measure generalisation.
  4. Compare the model against 4 fair baselines on each fold.
  5. Check the promotion gate (improvement ≥ 8%, coverage in [85%, 98%]).
  6. Train a final model on the full dataset (85% train, 15% val).
  7. Apply K5 consistency check: final MAE must not be >1.5× CV MAE.
  8. Serialise the model bundle and metadata to disk.

Key evaluation concepts:
  - Pinball loss: asymmetric loss used to evaluate quantile predictions.
  - Coverage: fraction of true values falling inside the [q03, q97] interval.
    Target: 85–98%.  Below 85% = too narrow intervals; above 98% = too wide.
  - Promotion gate: a model is only deployed ('promoted') if it clears both
    improvement and coverage thresholds.  This prevents deploying regressions.

Key symbols:
    TrainGBDTConfig: Run-level configuration (paths, zone, horizon, folds).
    evaluate_fold: Train + evaluate one CV fold, return metrics dict.
    train_final_model: Train on full data with 85/15 chronological split.
    run_full_pipeline: End-to-end training pipeline, returns result dict.
    save_gbdt_bundle / load_gbdt_bundle: Persistence helpers (also in model_store).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# Add the parent directory to sys.path so this script can be run directly
# (e.g., python train_gbdt.py --data training_data.parquet) without
# installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features_excel import (
    CAPACITY_TOTAL,
    FEATURE_SET_VERSION,
    SupervisedDataset,
    build_supervised_dataset,
    load_training_data,
    walk_forward_splits,
)
from model_gbdt import (
    GBDTBundle,
    GBDTConfig,
    predict_gbdt,
    train_gbdt_quantiles,
)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainGBDTConfig:
    """Top-level configuration for a GBDT training run.

    Attributes:
        model_dir: Directory where model bundles are saved.
            Bundle is written to model_dir/{zone_id}/h{horizon_minutes}/.
        zone_id: Identifier for the zone being modelled.  Multiple zones
            can have independent models in the same model_dir.
        horizon_steps: Forecast horizon in 15-min steps.
            4 = 60 min (H60, the primary production horizon).
        training_data_path: Absolute path to the training_data.parquet file
            produced by prepare_training_data.py.
        n_folds: Number of walk-forward CV folds.  6 is the default, giving
            ~5.5 weeks per fold when evaluated on Jul-Dec 2025.
        min_train_points: Minimum number of rows required to proceed.
            Fails early if the dataset is too small to train reliably.
        gbdt_config: LightGBM hyperparameters.  Defaults to GBDTConfig() if None.
    """
    model_dir: str = "models"
    zone_id: str = "default-zone"
    horizon_steps: int = 4          # 4 × 15 min = 60 min
    training_data_path: str = ""
    n_folds: int = 6
    min_train_points: int = 500
    gbdt_config: GBDTConfig | None = None


# ---------------------------------------------------------------------------
# Metric helper functions
# ---------------------------------------------------------------------------

def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Symmetric Mean Absolute Percentage Error (sMAPE).

    sMAPE is scale-free and handles near-zero values better than MAPE by
    normalising by the average of true and predicted values.

        sMAPE = mean(|y - f| / ((|y| + |f|) / 2)) × 100

    The denominator is clamped to 1e-6 to avoid division by zero when
    both y and f are near zero.

    Args:
        y_true: Ground-truth array.
        y_pred: Prediction array.

    Returns:
        sMAPE as a percentage (0-100+).
    """
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, 1e-6)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Compute the pinball loss for a single quantile.

    The pinball loss (also called quantile loss or check function) is the
    proper scoring rule for quantile regression:

        L_alpha(y, f) = alpha × max(y - f, 0) + (1 - alpha) × max(f - y, 0)

    Interpretation:
      - If alpha = 0.5: equivalent to half the mean absolute error (MAE).
      - If alpha = 0.97: the model is penalised 97× more for underestimation
        than overestimation, driving it to predict a high upper bound.
      - A perfect q97 model outputs the 97th percentile of the distribution.

    Args:
        y_true: Ground-truth array.
        y_pred: Predicted quantile values.
        alpha: Target quantile level in (0, 1).

    Returns:
        Mean pinball loss (non-negative float).
    """
    diff = y_true - y_pred
    return float(np.mean(np.where(diff >= 0, alpha * diff, (alpha - 1) * diff)))


def _coverage(y_true: np.ndarray, q_low: np.ndarray, q_high: np.ndarray) -> float:
    """Compute prediction interval coverage rate.

    Coverage measures what fraction of true values fall inside [q_low, q_high].
    For a 94% prediction interval (q03 to q97), the ideal coverage is 94%.
    In practice, coverage in [85%, 98%] is considered acceptable.

    Args:
        y_true: Ground-truth array.
        q_low: Lower quantile predictions (q03).
        q_high: Upper quantile predictions (q97).

    Returns:
        Coverage as a percentage (0-100).
    """
    covered = ((y_true >= q_low) & (y_true <= q_high)).astype(float)
    return float(np.mean(covered) * 100)


# ---------------------------------------------------------------------------
# Core training functions
# ---------------------------------------------------------------------------

def evaluate_fold(
    dataset: SupervisedDataset,
    fold: dict,
    config: TrainGBDTConfig,
) -> dict[str, Any]:
    """Train a fresh model on one fold and evaluate it against multiple baselines.

    A 'fold' is a dict produced by walk_forward_splits (features_excel.py).
    It contains chronologically ordered index lists: train_idx (past), val_idx
    (recent past for early stopping), test_idx (future, never seen during training).

    The model is evaluated against four baselines to demonstrate that the
    LightGBM model adds real predictive value beyond trivial reference forecasts:

      1. Persistence H60 (primary baseline): "Occupancy will be the same as it was
         60 min ago."  This is the fairest baseline for a 60-min forecast.
      2. Seasonal naive: "Occupancy will equal what it was yesterday at this hour."
      3. Rolling mean 1h: "Occupancy will equal the average of the last 4 readings."
      4. Global mean: "Occupancy will equal the training-set mean."

    The primary promotion baseline is Persistence H60.  The model must beat it by
    ≥ 8% (relative MAE improvement) to be considered deployable.

    Args:
        dataset: Full SupervisedDataset (all folds combined).
        fold: Dict from walk_forward_splits with train_idx, val_idx, test_idx.
        config: Training configuration (zone, hyperparameters, etc.).

    Returns:
        Dict containing all metrics for this fold, plus the trained bundle.
        The 'bundle' key is excluded when saving to metadata JSON.
    """
    gbdt_cfg = config.gbdt_config or GBDTConfig()

    # Extract split indices from the fold definition.
    train_idx = fold["train_idx"]
    val_idx = fold["val_idx"]
    test_idx = fold["test_idx"]

    # Slice the dataset into train / val / test matrices.
    # .values converts DataFrame to numpy array for LightGBM.
    X_train = dataset.X.iloc[train_idx].values.astype(np.float32)
    y_train = dataset.y[train_idx]
    X_val = dataset.X.iloc[val_idx].values.astype(np.float32)
    y_val = dataset.y[val_idx]
    X_test = dataset.X.iloc[test_idx].values.astype(np.float32)
    y_test = dataset.y[test_idx]

    # Train all three quantile models (q03, q50, q97) with early stopping on val.
    bundle = train_gbdt_quantiles(
        X_train, y_train, X_val, y_val,
        feature_columns=dataset.feature_columns,
        config=gbdt_cfg,
    )

    # Predict on the held-out test set (clipped to [0, CAPACITY_TOTAL]).
    preds = predict_gbdt(bundle, X_test, capacity=CAPACITY_TOTAL)

    # -----------------------------------------------------------------------
    # Baseline predictions — built from features already in X_test.
    # Using features from the test set (not future data) ensures baselines are
    # fair: they use only information available at prediction time.
    # -----------------------------------------------------------------------
    fc = dataset.feature_columns

    def _get_col(name: str) -> np.ndarray | None:
        """Extract a feature column from X_test and clip it to [0, capacity]."""
        if name in fc:
            return X_test[:, fc.index(name)].clip(0, CAPACITY_TOTAL)
        return None

    # Persistence H60: occupancy_lag_4 = the value from 4 × 15 min = 60 min ago.
    # This is the "do nothing" forecast: predict that occupancy stays constant.
    bl_persistence_h60 = _get_col("occupancy_lag_4")

    # Seasonal naive: occupancy_lag_day = value from ~12 operating hours ago
    # (same time slot yesterday in terms of opening hours).
    bl_seasonal = _get_col("occupancy_lag_day")

    # Rolling mean 1h: average of the last 4 readings (= last 60 min).
    bl_rolling_1h = _get_col("occupancy_roll_mean_4")

    # Global mean: constant prediction = mean of training labels.
    # Always available; serves as the weakest but most stable baseline.
    bl_global_mean = np.full(len(y_test), np.mean(y_train))

    # Use persistence as the primary baseline; fall back to global mean if unavailable.
    primary_baseline = bl_persistence_h60 if bl_persistence_h60 is not None else bl_global_mean

    # -----------------------------------------------------------------------
    # Compute evaluation metrics for GBDT (q50 = median prediction).
    # -----------------------------------------------------------------------
    mae_gbdt = float(np.mean(np.abs(y_test - preds["q50"])))
    rmse_gbdt = float(np.sqrt(np.mean((y_test - preds["q50"]) ** 2)))
    mdae_gbdt = float(np.median(np.abs(y_test - preds["q50"])))
    smape_gbdt = _smape(y_test, preds["q50"])

    # Baseline MAEs for comparison.
    mae_bl_persistence = float(np.mean(np.abs(y_test - primary_baseline)))
    mae_bl_seasonal = (
        float(np.mean(np.abs(y_test - bl_seasonal))) if bl_seasonal is not None else None
    )
    mae_bl_rolling = (
        float(np.mean(np.abs(y_test - bl_rolling_1h))) if bl_rolling_1h is not None else None
    )
    mae_bl_global = float(np.mean(np.abs(y_test - bl_global_mean)))

    # Quantile-specific metrics: pinball loss for each quantile.
    pinball_q03 = _pinball_loss(y_test, preds["q03"], 0.03)
    pinball_q50 = _pinball_loss(y_test, preds["q50"], 0.5)
    pinball_q97 = _pinball_loss(y_test, preds["q97"], 0.97)

    # Coverage: fraction of true values inside the [q03, q97] interval.
    coverage_94 = _coverage(y_test, preds["q03"], preds["q97"])

    # Relative improvement over the primary baseline (higher = better model).
    improvement = 0.0
    if mae_bl_persistence > 1e-9:
        improvement = float((mae_bl_persistence - mae_gbdt) / mae_bl_persistence * 100)

    return {
        "fold": fold["fold"],
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        # GBDT prediction quality metrics
        "mae_gbdt": mae_gbdt,
        "rmse_gbdt": rmse_gbdt,
        "mdae_gbdt": mdae_gbdt,
        "smape_gbdt": smape_gbdt,
        # Baseline comparison
        "mae_bl_persistence_h60": mae_bl_persistence,
        "mae_bl_seasonal": mae_bl_seasonal,
        "mae_bl_rolling_1h": mae_bl_rolling,
        "mae_bl_global_mean": mae_bl_global,
        "improvement_vs_persistence_pct": improvement,
        # Quantile metrics
        "pinball_q03": pinball_q03,
        "pinball_q50": pinball_q50,
        "pinball_q97": pinball_q97,
        "coverage_94": coverage_94,
        # The trained bundle is needed to aggregate feature importance later;
        # it is excluded from JSON serialisation (see run_full_pipeline).
        "feature_importance": bundle.feature_importance,
        "bundle": bundle,
    }


def train_final_model(
    dataset: SupervisedDataset,
    config: TrainGBDTConfig,
) -> tuple[GBDTBundle, dict[str, Any]]:
    """Train the final production model on the full dataset.

    After cross-validation confirms the model is good, we train one final model
    that uses all available data.  This final model is what gets deployed.

    Split strategy: 85% for training, 15% for validation (chronological split).
    The 15% validation set is used only for early stopping — it is NOT the same
    validation set as in the CV folds.  Using all data for the final model
    maximises the information available for production predictions.

    Args:
        dataset: Full SupervisedDataset (all available data).
        config: Training configuration.

    Returns:
        Tuple of (GBDTBundle, metrics_dict).
        metrics_dict contains mae_val, coverage_94_val, train_rows, val_rows.
    """
    gbdt_cfg = config.gbdt_config or GBDTConfig()
    n = len(dataset.X)
    # Chronological split: first 85% for training, last 15% for validation.
    # We do NOT shuffle because future data must not appear in training.
    train_end = int(n * 0.85)

    X_train = dataset.X.iloc[:train_end].values.astype(np.float32)
    y_train = dataset.y[:train_end]
    X_val = dataset.X.iloc[train_end:].values.astype(np.float32)
    y_val = dataset.y[train_end:]

    bundle = train_gbdt_quantiles(
        X_train, y_train, X_val, y_val,
        feature_columns=dataset.feature_columns,
        config=gbdt_cfg,
    )

    # Evaluate the final model on its validation portion to catch overfitting.
    preds = predict_gbdt(bundle, X_val, capacity=CAPACITY_TOTAL)
    mae = float(np.mean(np.abs(y_val - preds["q50"])))
    coverage = _coverage(y_val, preds["q03"], preds["q97"])

    metrics = {
        "mae_val": mae,
        "coverage_94_val": coverage,
        "train_rows": train_end,
        "val_rows": n - train_end,
    }
    return bundle, metrics


# ---------------------------------------------------------------------------
# Bundle persistence
# ---------------------------------------------------------------------------

def save_gbdt_bundle(
    bundle: GBDTBundle,
    model_dir: str,
    zone_id: str,
    horizon: int,
    metadata: dict[str, Any],
) -> dict[str, str]:
    """Persist the GBDT bundle to disk.

    The bundle is saved as five files in the bundle directory:
      - gbdt_q10.joblib: serialised LGBMRegressor at alpha=0.03
      - gbdt_q50.joblib: serialised LGBMRegressor at alpha=0.50
      - gbdt_q90.joblib: serialised LGBMRegressor at alpha=0.97
      - metadata_gbdt.json: training metrics, config, feature list, promoted flag
      - feature_importance.json: sorted feature → importance score mapping

    The directory structure is: model_dir/{zone_id}/h{horizon}/

    Args:
        bundle: Trained GBDTBundle.
        model_dir: Root directory for model storage.
        zone_id: Zone identifier (used as a subdirectory name).
        horizon: Forecast horizon in minutes (used in the directory name).
        metadata: Dict of metrics, config, version info to persist as JSON.

    Returns:
        Dict mapping file role → absolute path string.
    """
    from model_store import bundle_dir

    root = bundle_dir(model_dir, zone_id, horizon)
    root.mkdir(parents=True, exist_ok=True)

    # Serialise each model using joblib (standard Python ML serialisation).
    joblib.dump(bundle.model_q10, root / "gbdt_q10.joblib")
    joblib.dump(bundle.model_q50, root / "gbdt_q50.joblib")
    joblib.dump(bundle.model_q90, root / "gbdt_q90.joblib")

    # Persist metadata for auditability and reproducibility.
    (root / "metadata_gbdt.json").write_text(json.dumps(metadata, indent=2, default=str))

    # Feature importance is saved separately for easy inspection without loading
    # the full 3-model bundle.
    (root / "feature_importance.json").write_text(
        json.dumps(bundle.feature_importance, indent=2)
    )

    paths = {
        "root": str(root),
        "gbdt_q10": str(root / "gbdt_q10.joblib"),
        "gbdt_q50": str(root / "gbdt_q50.joblib"),
        "gbdt_q90": str(root / "gbdt_q90.joblib"),
        "metadata": str(root / "metadata_gbdt.json"),
        "feature_importance": str(root / "feature_importance.json"),
    }
    return paths


def load_gbdt_bundle(model_dir: str, zone_id: str, horizon: int) -> GBDTBundle:
    """Load a previously saved GBDT bundle from disk.

    Reconstructs the GBDTBundle dataclass from the three .joblib files and the
    metadata JSON.  The GBDTConfig is set to defaults (the exact hyperparameters
    used for training are stored in metadata_gbdt.json for reference).

    Args:
        model_dir: Root directory where bundles are stored.
        zone_id: Zone identifier.
        horizon: Forecast horizon in minutes (must match the saved bundle).

    Returns:
        GBDTBundle ready for predict_gbdt().
    """
    from model_store import bundle_dir

    root = bundle_dir(model_dir, zone_id, horizon)
    model_q10 = joblib.load(root / "gbdt_q10.joblib")
    model_q50 = joblib.load(root / "gbdt_q50.joblib")
    model_q90 = joblib.load(root / "gbdt_q90.joblib")

    metadata = json.loads((root / "metadata_gbdt.json").read_text())
    fi = json.loads((root / "feature_importance.json").read_text())

    return GBDTBundle(
        model_q10=model_q10,
        model_q50=model_q50,
        model_q90=model_q90,
        feature_columns=metadata.get("feature_columns", []),
        feature_importance=fi,
        config=GBDTConfig(),   # Hyperparameters are in metadata_gbdt.json for reference
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(config: TrainGBDTConfig) -> dict[str, Any]:
    """Execute the complete GBDT training and evaluation pipeline.

    This is the top-level function that ties together all training steps.
    It is called both from the CLI (main()) and from the forecast service's
    /v1/train endpoint when a retrain is triggered programmatically.

    Steps:
      1. Load the Parquet training data.
      2. Build the supervised dataset (X, y, timestamps).
      3. Run n-fold walk-forward CV; evaluate against 4 baselines per fold.
      4. Compute weighted aggregate metrics (weighted by test-set size per fold).
      5. Check the promotion gate:
           - Improvement vs Persistence H60 ≥ 8%  (model must beat the baseline)
           - Coverage94 in [85%, 98%]              (prediction intervals must be useful)
      6. Train the final model on the full dataset.
      7. K5 consistency check: final MAE ≤ 1.5× CV MAE (guards against overfitting).
      8. Serialise bundle + metadata; return result dict.

    Args:
        config: TrainGBDTConfig with paths, zone, horizon, and hyperparameters.

    Returns:
        Result dict containing: status, model_version, promoted flag, metrics,
        file paths, and top-10 feature importance.

    Raises:
        ValueError: If the training dataset has fewer rows than config.min_train_points.
    """
    print(f"[info] Loading training data from {config.training_data_path}")
    df = load_training_data(config.training_data_path)
    print(f"[info] Loaded {len(df)} rows")

    # Build supervised dataset: create target (y_tplus_{horizon_steps}) and align.
    print(f"[info] Building supervised dataset (horizon={config.horizon_steps} steps)")
    dataset = build_supervised_dataset(df, horizon_steps=config.horizon_steps)
    print(f"[info] Dataset: {len(dataset.X)} samples, {len(dataset.feature_columns)} features")

    # Guard against undersized datasets that would produce unreliable metrics.
    if len(dataset.X) < config.min_train_points:
        raise ValueError(
            f"Insufficient data: {len(dataset.X)} rows < {config.min_train_points} minimum"
        )

    # -----------------------------------------------------------------------
    # Step 3: Walk-forward cross-validation
    # -----------------------------------------------------------------------
    print(f"\n[info] Running {config.n_folds}-fold walk-forward cross-validation...")
    folds = walk_forward_splits(dataset, n_folds=config.n_folds)
    fold_results = []

    for fold in folds:
        print(f"\n--- Fold {fold['fold']} ---")
        if "train_range" in fold:
            print(f"  Train: {fold['train_range']}")
            print(f"  Test:  {fold['test_range']}")
        result = evaluate_fold(dataset, fold, config)
        fold_results.append(result)
        print(f"  MAE GBDT:              {result['mae_gbdt']:.3f}")
        print(f"  MAE Persistence H60:   {result['mae_bl_persistence_h60']:.3f}")
        if result.get("mae_bl_seasonal") is not None:
            print(f"  MAE Seasonal (yesterday): {result['mae_bl_seasonal']:.3f}")
        if result.get("mae_bl_rolling_1h") is not None:
            print(f"  MAE Rolling Mean 1h:   {result['mae_bl_rolling_1h']:.3f}")
        print(f"  Improvement vs Pers.:  {result['improvement_vs_persistence_pct']:.1f}%")
        print(f"  Coverage94:            {result['coverage_94']:.1f}%")

    # -----------------------------------------------------------------------
    # Step 4: Aggregate CV metrics weighted by test-set size.
    # Using weighted averages rather than simple averages is fairer because folds
    # with more test data should contribute more to the overall estimate.
    # -----------------------------------------------------------------------
    total_test = sum(r["test_size"] for r in fold_results)
    w_mae = sum(r["mae_gbdt"] * r["test_size"] for r in fold_results) / total_test
    w_bl_pers = sum(r["mae_bl_persistence_h60"] * r["test_size"] for r in fold_results) / total_test

    # Aggregate secondary baseline MAEs only if all folds have them (some may be None
    # if a feature column was absent from that fold's test set).
    w_bl_seasonal = None
    if all(r.get("mae_bl_seasonal") is not None for r in fold_results):
        w_bl_seasonal = sum(
            r["mae_bl_seasonal"] * r["test_size"] for r in fold_results
        ) / total_test
    w_bl_rolling = None
    if all(r.get("mae_bl_rolling_1h") is not None for r in fold_results):
        w_bl_rolling = sum(
            r["mae_bl_rolling_1h"] * r["test_size"] for r in fold_results
        ) / total_test
    w_coverage = sum(r["coverage_94"] * r["test_size"] for r in fold_results) / total_test

    # Compute improvement from aggregated MAEs rather than averaging per-fold
    # improvements.  Averaging percentages would be mathematically incorrect.
    improvement_vs_persistence = 0.0
    if w_bl_pers > 1e-9:
        improvement_vs_persistence = float((w_bl_pers - w_mae) / w_bl_pers * 100)

    # Standard deviation of fold MAEs: a measure of training stability.
    # High std (>15%) indicates the model is sensitive to which time period it trains on.
    fold_maes = [r["mae_gbdt"] for r in fold_results]
    mae_std = float(np.std(fold_maes))

    print(f"\n=== CROSS-VALIDATION SUMMARY (weighted by test size) ===")
    print(f"Weighted MAE GBDT:          {w_mae:.3f} (std across folds: {mae_std:.3f})")
    print(f"Weighted MAE Persistence:   {w_bl_pers:.3f}")
    if w_bl_seasonal is not None:
        print(f"Weighted MAE Seasonal:      {w_bl_seasonal:.3f}")
    if w_bl_rolling is not None:
        print(f"Weighted MAE Rolling 1h:    {w_bl_rolling:.3f}")
    print(f"Improvement vs Persistence: {improvement_vs_persistence:.1f}%")
    print(f"Weighted Coverage94:        {w_coverage:.1f}%")

    # -----------------------------------------------------------------------
    # Step 5: Promotion gate.
    # The model is only 'promoted' (deployable) if it clears both gates:
    #   1. Improves MAE over Persistence H60 by at least 8% (relative).
    #   2. Coverage of [q03, q97] interval falls in [85%, 98%].
    #      Below 85%: intervals are too narrow (overconfident).
    #      Above 98%: intervals are too wide (uninformative, vacuous).
    # -----------------------------------------------------------------------
    promoted = improvement_vs_persistence >= 8.0 and 85 <= w_coverage <= 98
    print(f"Promotion gate:             {'PASS' if promoted else 'FAIL'}")
    if not promoted:
        if improvement_vs_persistence < 8.0:
            print(f"  [blocked] Improvement {improvement_vs_persistence:.1f}% < 8% threshold")
        if not (85 <= w_coverage <= 98):
            print(f"  [blocked] Coverage {w_coverage:.1f}% outside [85%, 98%]")

    # -----------------------------------------------------------------------
    # Step 6: Train the final model on all available data.
    # -----------------------------------------------------------------------
    print(f"\n[info] Training final model on full dataset...")
    bundle, final_metrics = train_final_model(dataset, config)
    print(f"  Final MAE (val): {final_metrics['mae_val']:.3f}")
    print(f"  Final Coverage94 (val): {final_metrics['coverage_94_val']:.1f}%")

    # -----------------------------------------------------------------------
    # Step 7: K5 consistency check.
    # The final model's validation MAE must not be more than 1.5× the CV MAE.
    # A large gap indicates the final model overfit to training data in a way
    # that the CV evaluation did not detect.
    # -----------------------------------------------------------------------
    final_promoted = promoted
    if final_metrics["mae_val"] > w_mae * 1.5:
        print(f"  [warn] Final model MAE {final_metrics['mae_val']:.3f} > 1.5x CV MAE {w_mae:.3f}")
        final_promoted = False

    # -----------------------------------------------------------------------
    # Step 8: Build metadata record and save bundle.
    # The metadata JSON is the authoritative record of what was trained and how.
    # -----------------------------------------------------------------------
    horizon_minutes = config.horizon_steps * 15
    model_version = f"lgbm-quantile-v1-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    metadata = {
        "model_version": model_version,
        "backend": "lgbm",
        "product": "short_term",
        "zone_id": config.zone_id,
        "horizon": horizon_minutes,
        "horizon_steps": config.horizon_steps,
        "trained_at": datetime.now(UTC).isoformat(),
        "feature_set_version": FEATURE_SET_VERSION,
        "feature_columns": dataset.feature_columns,
        "target_name": dataset.target_name,
        "promoted": final_promoted,
        "scientific_status": "walk_forward_validated",
        "training_data": config.training_data_path,
        "quantile_levels": {"lower": 0.03, "median": 0.50, "upper": 0.97},
        "metrics": {
            "cv_weighted_mae": w_mae,
            "cv_mae_std": mae_std,
            "cv_weighted_mae_bl_persistence_h60": w_bl_pers,
            "cv_weighted_mae_bl_seasonal": w_bl_seasonal,
            "cv_weighted_mae_bl_rolling_1h": w_bl_rolling,
            "cv_improvement_vs_persistence_pct": improvement_vs_persistence,
            "cv_weighted_coverage_94": w_coverage,
            "final_mae_val": final_metrics["mae_val"],
            "final_coverage_94_val": final_metrics["coverage_94_val"],
        },
        # Per-fold details (bundle objects excluded via dict comprehension).
        "fold_results": [
            {k: v for k, v in r.items() if k != "bundle"}
            for r in fold_results
        ],
        "gbdt_config": {
            "num_leaves": (config.gbdt_config or GBDTConfig()).num_leaves,
            "learning_rate": (config.gbdt_config or GBDTConfig()).learning_rate,
            "n_estimators": (config.gbdt_config or GBDTConfig()).n_estimators,
        },
    }

    paths = save_gbdt_bundle(bundle, config.model_dir, config.zone_id, horizon_minutes, metadata)
    print(f"\n[done] Model saved to {paths['root']}")

    return {
        "status": "ok",
        "model_version": model_version,
        "promoted": final_promoted,
        "metrics": metadata["metrics"],
        "paths": paths,
        # Top-10 features by importance — useful for quick inspection.
        "feature_importance_top10": dict(list(bundle.feature_importance.items())[:10]),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Command-line interface for triggering a training run.

    Example usage:
        python train_gbdt.py --data training_data.parquet --model-dir ./models

    Returns:
        0 on success (used as shell exit code).
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train GBDT forecast model")
    parser.add_argument("--data", required=True, help="Path to training Parquet file")
    parser.add_argument("--model-dir", default="models", help="Model output directory")
    parser.add_argument("--zone-id", default=os.getenv("DEFAULT_ZONE_ID", "default-zone"))
    parser.add_argument("--horizon-steps", type=int, default=4,
                        help="Forecast horizon in 15-min steps (4 = 60 min)")
    parser.add_argument("--n-folds", type=int, default=6, help="Walk-forward CV folds")
    parser.add_argument("--num-leaves", type=int, default=31,
                        help="LightGBM num_leaves (tree complexity)")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-estimators", type=int, default=500,
                        help="Maximum boosting rounds (early stopping may terminate sooner)")
    args = parser.parse_args()

    gbdt_cfg = GBDTConfig(
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
    )
    config = TrainGBDTConfig(
        model_dir=args.model_dir,
        zone_id=args.zone_id,
        horizon_steps=args.horizon_steps,
        training_data_path=args.data,
        n_folds=args.n_folds,
        gbdt_config=gbdt_cfg,
    )

    result = run_full_pipeline(config)
    # Print the result as JSON for easy pipeline integration.
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
