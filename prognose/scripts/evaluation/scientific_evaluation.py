"""Scientific evaluation of the GBDT occupancy forecast model.

Multi-baseline comparison, Bootstrap confidence intervals,
Diebold-Mariano tests, segment analysis, residual diagnostics.

Usage:
    python3 scripts/evaluation/scientific_evaluation.py \
        --data training_data.parquet \
        --model-dir services/forecast/models
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib

FORECAST_DIR = Path(__file__).resolve().parent.parent.parent / "services" / "forecast"
sys.path.insert(0, str(FORECAST_DIR))

from features_excel import (
    CAPACITY_TOTAL,
    build_supervised_dataset,
    load_training_data,
    walk_forward_splits,
)
from model_gbdt import GBDTBundle, GBDTConfig, predict_gbdt, train_gbdt_quantiles
from model_store import bundle_dir


# =========================================================================
# Metrics
# =========================================================================

def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))

def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))

def mdae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.median(np.abs(y - p)))

def mase(y: np.ndarray, p: np.ndarray, naive_mae: float) -> float:
    if naive_mae < 1e-9:
        return float("inf")
    return mae(y, p) / naive_mae

def pinball(y: np.ndarray, p: np.ndarray, alpha: float) -> float:
    diff = y - p
    return float(np.mean(np.where(diff >= 0, alpha * diff, (alpha - 1) * diff)))

def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)) * 100)

def mean_interval_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(hi - lo))


# =========================================================================
# Statistical tests
# =========================================================================

def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float, float]:
    """Bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(42)
    means = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(np.mean(values)), lo, hi


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple[float, float]:
    """Diebold-Mariano test for equal predictive accuracy.

    H0: E[d_t] = 0 where d_t = e1_t^2 - e2_t^2.
    Returns (DM statistic, two-sided p-value).
    Positive DM => model 2 is better.
    """
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    d_bar = np.mean(d)

    # Newey-West variance estimator
    gamma_0 = np.mean((d - d_bar) ** 2)
    gamma_sum = 0.0
    for k in range(1, h):
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        gamma_sum += 2 * gamma_k
    var_d = (gamma_0 + gamma_sum) / n

    if var_d < 1e-12:
        return 0.0, 1.0

    dm_stat = d_bar / np.sqrt(var_d)

    # Two-sided p-value using normal approximation
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)


# =========================================================================
# Baseline models
# =========================================================================

def compute_baselines(X_test: np.ndarray, y_train: np.ndarray,
                      feature_columns: list[str]) -> dict[str, np.ndarray]:
    """Compute baseline predictions."""
    n = len(X_test)

    def _col(name: str) -> np.ndarray | None:
        if name in feature_columns:
            return X_test[:, feature_columns.index(name)].clip(0, CAPACITY_TOTAL)
        return None

    baselines = {}

    # 1. Persistence H60: value 60 min ago
    bl = _col("occupancy_lag_4")
    if bl is not None:
        baselines["Persistence H60"] = bl

    # 2. Seasonal naive: same time yesterday
    bl = _col("occupancy_lag_day")
    if bl is not None:
        baselines["Seasonal (yesterday)"] = bl

    # 3. Rolling mean 1h
    bl = _col("occupancy_roll_mean_4")
    if bl is not None:
        baselines["Rolling Mean 1h"] = bl

    # 4. Rolling mean 4h
    bl = _col("occupancy_roll_mean_16")
    if bl is not None:
        baselines["Rolling Mean 4h"] = bl

    # 5. Global mean
    baselines["Global Mean"] = np.full(n, np.mean(y_train))

    return baselines


# =========================================================================
# Segment analysis
# =========================================================================

def segment_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.Series,
    X: np.ndarray,
    feature_columns: list[str],
) -> dict[str, Any]:
    """MAE breakdown by hour, weekday, month, and load level."""
    ts = pd.to_datetime(timestamps.values)
    errors = np.abs(y_true - y_pred)

    segments: dict[str, Any] = {}

    # By hour
    hours = ts.hour
    by_hour = {}
    for h in sorted(set(hours)):
        mask = hours == h
        by_hour[str(h)] = {"mae": float(np.mean(errors[mask])), "n": int(mask.sum())}
    segments["by_hour"] = by_hour

    # By weekday
    weekdays = ts.weekday
    dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    by_dow = {}
    for d in sorted(set(weekdays)):
        mask = weekdays == d
        by_dow[dow_names.get(d, str(d))] = {"mae": float(np.mean(errors[mask])), "n": int(mask.sum())}
    segments["by_weekday"] = by_dow

    # By month
    months = ts.month
    by_month = {}
    for m in sorted(set(months)):
        mask = months == m
        by_month[str(m)] = {"mae": float(np.mean(errors[mask])), "n": int(mask.sum())}
    segments["by_month"] = by_month

    # By load level (based on actual occupancy)
    load_bins = [(0, 5, "very_low"), (5, 15, "low"), (15, 30, "medium"), (30, 50, "high"), (50, 84, "very_high")]
    by_load = {}
    for lo, hi, label in load_bins:
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() > 0:
            by_load[label] = {"mae": float(np.mean(errors[mask])), "n": int(mask.sum()),
                              "range": f"[{lo}, {hi})"}
    segments["by_load_level"] = by_load

    return segments


# =========================================================================
# Residual analysis
# =========================================================================

def residual_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute residual statistics."""
    residuals = y_true - y_pred
    return {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "median": float(np.median(residuals)),
        "skewness": float(pd.Series(residuals).skew()),
        "kurtosis": float(pd.Series(residuals).kurtosis()),
        "pct_positive": float(np.mean(residuals > 0) * 100),
        "pct_negative": float(np.mean(residuals < 0) * 100),
        "pct_zero": float(np.mean(residuals == 0) * 100),
        "max_overpredict": float(np.min(residuals)),
        "max_underpredict": float(np.max(residuals)),
        "p5": float(np.percentile(residuals, 5)),
        "p95": float(np.percentile(residuals, 95)),
    }


# =========================================================================
# Main evaluation
# =========================================================================

def run_scientific_evaluation(
    parquet_path: str,
    model_dir: str,
    zone_id: str = "default-zone",
    horizon: int = 60,
    n_folds: int = 6,
) -> dict[str, Any]:
    """Run the full scientific evaluation."""

    print("=" * 70)
    print("SCIENTIFIC EVALUATION -- GBDT Occupancy Forecast")
    print("=" * 70)

    # Load data and build dataset
    df = load_training_data(parquet_path)
    dataset = build_supervised_dataset(df, horizon_steps=horizon // 15)
    folds = walk_forward_splits(dataset, n_folds=n_folds)

    print(f"Data: {len(dataset.X)} samples, {len(dataset.feature_columns)} features")
    print(f"Folds: {len(folds)}")

    # Collect all out-of-sample predictions
    all_y_test = []
    all_preds_q50 = []
    all_preds_q03 = []
    all_preds_q97 = []
    all_baselines: dict[str, list] = {}
    all_timestamps = []

    for fold in folds:
        train_idx = fold["train_idx"]
        val_idx = fold["val_idx"]
        test_idx = fold["test_idx"]

        X_train = dataset.X.iloc[train_idx].values.astype(np.float32)
        y_train = dataset.y[train_idx]
        X_val = dataset.X.iloc[val_idx].values.astype(np.float32)
        y_val = dataset.y[val_idx]
        X_test = dataset.X.iloc[test_idx].values.astype(np.float32)
        y_test = dataset.y[test_idx]

        # Train model
        bundle = train_gbdt_quantiles(
            X_train, y_train, X_val, y_val,
            feature_columns=dataset.feature_columns,
            config=GBDTConfig(),
        )
        preds = predict_gbdt(bundle, X_test, capacity=CAPACITY_TOTAL)

        all_y_test.append(y_test)
        all_preds_q50.append(preds["q50"])
        all_preds_q03.append(preds["q03"])
        all_preds_q97.append(preds["q97"])
        all_timestamps.append(dataset.timestamps.iloc[test_idx])

        # Baselines
        bl = compute_baselines(X_test, y_train, dataset.feature_columns)
        for name, vals in bl.items():
            all_baselines.setdefault(name, []).append(vals)

        print(f"  Fold {fold['fold']}: test={len(y_test)}, MAE={mae(y_test, preds['q50']):.3f}")

    # Concatenate all OOS predictions
    y_all = np.concatenate(all_y_test)
    p50_all = np.concatenate(all_preds_q50)
    p03_all = np.concatenate(all_preds_q03)
    p97_all = np.concatenate(all_preds_q97)
    ts_all = pd.concat(all_timestamps, ignore_index=True)

    print(f"\nTotal OOS samples: {len(y_all)}")

    # =====================================================================
    # 1. Point metrics -- GBDT
    # =====================================================================
    print("\n--- 1. POINT METRICS (GBDT) ---")
    e_gbdt = y_all - p50_all
    gbdt_metrics = {
        "mae": mae(y_all, p50_all),
        "rmse": rmse(y_all, p50_all),
        "mdae": mdae(y_all, p50_all),
    }
    print(f"  MAE:  {gbdt_metrics['mae']:.3f}")
    print(f"  RMSE: {gbdt_metrics['rmse']:.3f}")
    print(f"  MdAE: {gbdt_metrics['mdae']:.3f}")

    # =====================================================================
    # 2. Multi-baseline comparison
    # =====================================================================
    print("\n--- 2. MULTI-BASELINE COMPARISON ---")
    baseline_metrics = {}
    for name in all_baselines:
        bl_all = np.concatenate(all_baselines[name])
        bl_mae = mae(y_all, bl_all)
        bl_rmse = rmse(y_all, bl_all)
        e_bl = y_all - bl_all
        improvement = (bl_mae - gbdt_metrics["mae"]) / bl_mae * 100 if bl_mae > 1e-9 else 0

        # MASE: MAE / naive_MAE
        mase_val = gbdt_metrics["mae"] / bl_mae if bl_mae > 1e-9 else float("inf")

        # Diebold-Mariano test
        dm_stat, dm_p = diebold_mariano(e_gbdt, e_bl, h=4)

        baseline_metrics[name] = {
            "mae": bl_mae,
            "rmse": bl_rmse,
            "improvement_pct": improvement,
            "mase": mase_val,
            "dm_statistic": dm_stat,
            "dm_p_value": dm_p,
            "dm_significant_5pct": dm_p < 0.05,
        }
        sig = "*" if dm_p < 0.05 else ""
        print(f"  {name:25s}  MAE={bl_mae:.3f}  Impr={improvement:+.1f}%  "
              f"MASE={mase_val:.3f}  DM={dm_stat:+.2f} (p={dm_p:.4f}){sig}")

    # =====================================================================
    # 3. Bootstrap CI for GBDT MAE
    # =====================================================================
    print("\n--- 3. BOOTSTRAP CONFIDENCE INTERVALS ---")
    abs_errors = np.abs(e_gbdt)
    mae_mean, mae_lo, mae_hi = bootstrap_ci(abs_errors, n_boot=2000)
    print(f"  MAE = {mae_mean:.3f}  95% CI: [{mae_lo:.3f}, {mae_hi:.3f}]")

    # Bootstrap CI for improvement vs Persistence H60
    if "Persistence H60" in all_baselines:
        bl_pers = np.concatenate(all_baselines["Persistence H60"])
        bl_errors = np.abs(y_all - bl_pers)
        diff_errors = bl_errors - abs_errors  # positive = GBDT is better
        diff_mean, diff_lo, diff_hi = bootstrap_ci(diff_errors, n_boot=2000)
        print(f"  MAE diff (Persistence - GBDT) = {diff_mean:.3f}  95% CI: [{diff_lo:.3f}, {diff_hi:.3f}]")
        print(f"  {'GBDT significantly better' if diff_lo > 0 else 'Improvement CI includes 0!'}")

    # =====================================================================
    # 4. Quantile / Interval metrics
    # =====================================================================
    print("\n--- 4. QUANTILE / INTERVAL METRICS ---")
    cov94 = coverage(y_all, p03_all, p97_all)
    pb03 = pinball(y_all, p03_all, 0.03)
    pb50 = pinball(y_all, p50_all, 0.50)
    pb97 = pinball(y_all, p97_all, 0.97)
    miw = mean_interval_width(p03_all, p97_all)

    quantile_metrics = {
        "coverage_94": cov94,
        "pinball_q03": pb03,
        "pinball_q50": pb50,
        "pinball_q97": pb97,
        "mean_interval_width": miw,
    }
    print(f"  Coverage (94% target): {cov94:.1f}%")
    print(f"  Pinball q03: {pb03:.4f}  q50: {pb50:.4f}  q97: {pb97:.4f}")
    print(f"  Mean interval width: {miw:.2f}")

    calibration_ok = 85 <= cov94 <= 98
    print(f"  Calibration: {'OK' if calibration_ok else 'OUT OF RANGE'} "
          f"(target: [85%, 98%])")

    # =====================================================================
    # 5. Segment analysis
    # =====================================================================
    print("\n--- 5. SEGMENT ANALYSIS ---")
    X_oos = np.concatenate([
        dataset.X.iloc[fold["test_idx"]].values for fold in folds
    ])
    segments = segment_analysis(y_all, p50_all, ts_all, X_oos, dataset.feature_columns)

    print("  By hour (MAE):")
    for h, v in segments["by_hour"].items():
        bar = "█" * int(v["mae"] * 10)
        print(f"    {h:>2}h: {v['mae']:.3f} (n={v['n']:>4d}) {bar}")

    print("  By weekday (MAE):")
    for d, v in segments["by_weekday"].items():
        print(f"    {d}: {v['mae']:.3f} (n={v['n']:>4d})")

    print("  By month (MAE):")
    for m, v in segments["by_month"].items():
        print(f"    Month {m:>2s}: {v['mae']:.3f} (n={v['n']:>4d})")

    print("  By load level (MAE):")
    for level, v in segments["by_load_level"].items():
        print(f"    {level:>10s} {v['range']}: {v['mae']:.3f} (n={v['n']:>4d})")

    # =====================================================================
    # 6. Residual analysis
    # =====================================================================
    print("\n--- 6. RESIDUAL ANALYSIS ---")
    res = residual_analysis(y_all, p50_all)
    print(f"  Mean residual (bias):   {res['mean']:+.3f}")
    print(f"  Std residual:           {res['std']:.3f}")
    print(f"  Skewness:               {res['skewness']:+.3f}")
    print(f"  Kurtosis:               {res['kurtosis']:.3f}")
    print(f"  % overpredicting:       {res['pct_negative']:.1f}%")
    print(f"  % underpredicting:      {res['pct_positive']:.1f}%")
    print(f"  Max overpredict:        {res['max_overpredict']:.1f}")
    print(f"  Max underpredict:       {res['max_underpredict']:.1f}")
    print(f"  P5/P95 residuals:       [{res['p5']:.1f}, {res['p95']:.1f}]")

    # =====================================================================
    # 7. Fold stability
    # =====================================================================
    print("\n--- 7. FOLD STABILITY ---")
    fold_maes = []
    for i, fold in enumerate(folds):
        y_f = all_y_test[i]
        p_f = all_preds_q50[i]
        f_mae = mae(y_f, p_f)
        fold_maes.append(f_mae)
        print(f"  Fold {fold['fold']}: MAE={f_mae:.3f}  n={len(y_f)}")

    fold_mae_arr = np.array(fold_maes)
    print(f"  Mean: {fold_mae_arr.mean():.3f}  Std: {fold_mae_arr.std():.3f}  "
          f"CV: {fold_mae_arr.std() / fold_mae_arr.mean() * 100:.1f}%")

    # =====================================================================
    # 8. GO/NO-GO assessment
    # =====================================================================
    print("\n" + "=" * 70)
    print("GO/NO-GO ASSESSMENT")
    print("=" * 70)

    checks = {
        "MAE < best baseline": gbdt_metrics["mae"] < min(bm["mae"] for bm in baseline_metrics.values()),
        "Improvement >= 8% vs Persistence": baseline_metrics.get("Persistence H60", {}).get("improvement_pct", 0) >= 8,
        "Coverage94 in [85%, 98%]": 85 <= cov94 <= 98,
        "DM significant vs Persistence (p<0.05)": baseline_metrics.get("Persistence H60", {}).get("dm_significant_5pct", False),
        "Bootstrap CI excludes 0 improvement": diff_lo > 0 if "Persistence H60" in all_baselines else False,
        "Fold CV < 15%": fold_mae_arr.std() / fold_mae_arr.mean() < 0.15,
        "Mean residual (bias) < 0.5": abs(res["mean"]) < 0.5,
        "No extreme segment MAE (< 5.0)": all(v["mae"] < 5.0 for v in segments["by_hour"].values()),
    }

    all_pass = True
    for check_name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {check_name}")

    print(f"\nOVERALL: {'GO' if all_pass else 'NO-GO'}")

    # =====================================================================
    # Build report
    # =====================================================================
    report = {
        "model_version": "lgbm-quantile-v1",
        "feature_set": "excel_v1",
        "horizon_minutes": horizon,
        "n_folds": len(folds),
        "total_oos_samples": len(y_all),
        "gbdt_metrics": gbdt_metrics,
        "baseline_metrics": baseline_metrics,
        "bootstrap_ci_mae": {"mean": mae_mean, "lo_95": mae_lo, "hi_95": mae_hi},
        "quantile_metrics": quantile_metrics,
        "segments": segments,
        "residuals": res,
        "fold_stability": {
            "fold_maes": fold_maes,
            "mean": float(fold_mae_arr.mean()),
            "std": float(fold_mae_arr.std()),
            "cv_pct": float(fold_mae_arr.std() / fold_mae_arr.mean() * 100),
        },
        "go_no_go": {k: v for k, v in checks.items()},
        "overall_verdict": "GO" if all_pass else "NO-GO",
    }

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Scientific evaluation of GBDT forecast model")
    parser.add_argument("--data", required=True, help="Path to training Parquet")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--zone-id", default="default-zone")
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--n-folds", type=int, default=6)
    args = parser.parse_args()

    report = run_scientific_evaluation(
        args.data, args.model_dir, args.zone_id, args.horizon, args.n_folds,
    )

    # Save report
    report_path = Path(args.model_dir) / "scientific_evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport saved to: {report_path}")

    return 0 if report["overall_verdict"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
