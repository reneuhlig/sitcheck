"""Scientific evaluation framework for occupancy forecasting models.

This module implements a rigorous multi-model evaluation methodology for assessing
whether a trained forecast model is better than naive baselines, and whether it
is statistically significantly better using the Diebold-Mariano test.

Evaluation methodology:
  - Rolling-origin cross-validation: instead of a single train/test split, many
    forecast origins are sampled across the evaluation period.  At each origin, the
    model is presented with history up to that point and asked to forecast ahead.
    This simulates the actual deployment scenario (predict from current time) and
    yields a large number of independent forecast errors for statistical testing.
  - Multiple baseline comparisons: persistence, seasonal naive, rolling mean, global
    mean, and a linear regression baseline.
  - Segment analysis: metrics are broken down by lecture-activity segments (active
    lectures, heavy load, transition periods) to identify situational weaknesses.
  - Diebold-Mariano test: formal statistical test for whether the model's forecast
    errors are significantly smaller than the baseline's.  p < 0.05 indicates the
    improvement is unlikely to be due to chance.

Key symbols:
    EvaluationConfig: All parameters controlling the evaluation run.
    evaluate_training_run: Top-level function — run full evaluation, return report dict.
    save_evaluation_report / load_evaluation_report: JSON persistence for reports.
    diebold_mariano_test: Formal hypothesis test for forecast improvement.
    filter_history_for_scientific_eval: Remove low-quality sensor rows before eval.
"""
from __future__ import annotations

import json
import math
import re
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from features import build_feature_frame, build_inference_vector
from model_store import load_bundle

# Quality flags that indicate sensor malfunction or data corruption.
# Rows with these flags are excluded from evaluation to avoid penalising the
# model for impossible-to-predict sensor anomalies.
DEFAULT_HARD_QUALITY_FLAGS = [
    "TRACK_ERROR",
    "SERIALIZATION_ERROR",
    "BACKLOG_OVERFLOW",
    "ZONE_MISSING",
]

MODEL_BASELINE = "baseline"
MODEL_TF = "tf_mlp"
MODEL_GBDT = "quantile_gbdt"
MODEL_SARIMAX = "sarimax"

SEGMENT_LECTURE_ACTIVE = "lecture_active"
SEGMENT_HEAVY_EFFECT = "heavy_effect"
SEGMENT_LECTURE_TRANSITION_START = "lecture_transition_start"
SEGMENT_LECTURE_TRANSITION_END = "lecture_transition_end"

SEGMENT_FIELDS = {
    SEGMENT_LECTURE_ACTIVE: "segment_lecture_active",
    SEGMENT_HEAVY_EFFECT: "segment_heavy_effect",
    SEGMENT_LECTURE_TRANSITION_START: "segment_lecture_transition_start",
    SEGMENT_LECTURE_TRANSITION_END: "segment_lecture_transition_end",
}


@dataclass
class EvaluationConfig:
    """Configuration for a scientific evaluation run.

    Controls the rolling-origin CV structure, data quality filtering, model
    backends to evaluate, promotion gate thresholds, and segment analysis.

    Attributes:
        zone_id: Zone being evaluated.
        horizons: List of forecast horizons in minutes to evaluate (e.g. [60, 240]).
        folds: Number of time-based train/val/test splits to evaluate over.
        train_days: Length of training window per fold (days).
        val_days: Length of validation window per fold (days).
        test_days: Length of test window per fold (days).
        gap_minutes: Gap between train end and test start (prevents lag leakage).
        origin_stride_minutes: How often to sample a new forecast origin within the
            test window.  60 = one origin per hour.
        max_origins_per_split: Cap on origins per fold to limit compute time.
        min_quality_score: Rows below this score are excluded from evaluation.
        exclude_quality_flags: Rows with these sensor error flags are excluded.
        random_seed: Seed for reproducibility.
        tf_epochs: Number of TF-MLP training epochs (kept low for eval speed).
        tf_batch_size: Mini-batch size for TF training within each fold.
        tf_min_train_points: Minimum rows for TF training; skip fold if below.
        tf_max_horizon_minutes: Skip TF evaluation beyond this horizon (slow).
        enable_tf: Deprecated. TF-MLP is removed from active evaluation.
        enable_sarimax: Whether to include SARIMAX in the evaluation.
        improvement_threshold: Minimum relative MAE improvement for promotion (0.08 = 8%).
        max_long_degradation: Maximum allowed degradation at long horizons (0.02 = 2%).
        coverage_low / coverage_high: Acceptable range for PI coverage (0.85–0.95).
        primary_horizon: The horizon used for the main promotion decision (minutes).
        include_lecture_impact: Whether to pass lecture features to models during eval.
        segment_min_samples: Minimum samples for a segment metric to be reported.
    """
    zone_id: str
    horizons: list[int]
    folds: int = 6
    train_days: int = 30
    val_days: int = 7
    test_days: int = 7
    gap_minutes: int = 60
    origin_stride_minutes: int = 60
    max_origins_per_split: int = 120
    min_quality_score: float = 0.6
    exclude_quality_flags: list[str] = field(default_factory=lambda: list(DEFAULT_HARD_QUALITY_FLAGS))
    random_seed: int = 42
    tf_epochs: int = 16
    tf_batch_size: int = 64
    tf_min_train_points: int = 400
    tf_max_horizon_minutes: int = 720
    enable_tf: bool = False
    enable_sarimax: bool = True
    improvement_threshold: float = 0.08
    max_long_degradation: float = 0.02
    coverage_low: float = 0.85
    coverage_high: float = 0.95
    primary_horizon: int = 60
    include_lecture_impact: bool = True
    segment_min_samples: int = 30


def _safe_name(value: str) -> str:
    """Sanitise a string for use as a filesystem path component."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "default"


def report_dir(model_dir: str, zone_id: str) -> Path:
    """Return the directory where evaluation reports are stored for a zone."""
    return Path(model_dir) / "reports" / _safe_name(zone_id)


def save_evaluation_report(model_dir: str, zone_id: str, report: dict[str, Any]) -> dict[str, str]:
    """Save an evaluation report to disk as JSON.

    Writes two files: a uniquely named run file ({run_id}.json) and a
    'latest.json' symlink-equivalent so callers can always find the most recent
    report without knowing the run ID.

    Args:
        model_dir: Root model directory.
        zone_id: Zone identifier.
        report: Evaluation report dict (will have 'run_id' injected if absent).

    Returns:
        Dict with run_id, path (unique run file), and latest (overwritten each run).
    """
    root = report_dir(model_dir=model_dir, zone_id=zone_id)
    root.mkdir(parents=True, exist_ok=True)
    run_id = str(
        report.get("run_id")
        or f"eval-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    )
    report["run_id"] = run_id

    run_path = root / f"{run_id}.json"
    latest_path = root / "latest.json"

    run_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return {
        "run_id": run_id,
        "path": str(run_path),
        "latest": str(latest_path),
    }


def load_evaluation_report(model_dir: str, zone_id: str, run_id: str) -> dict[str, Any] | None:
    """Load a specific evaluation report by run ID.  Returns None if not found."""
    path = report_dir(model_dir=model_dir, zone_id=zone_id) / f"{run_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_latest_evaluation_report(model_dir: str, zone_id: str) -> dict[str, Any] | None:
    """Load the most recent evaluation report.  Returns None if no report exists."""
    path = report_dir(model_dir=model_dir, zone_id=zone_id) / "latest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _quality_flag_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    flags: set[str] = set()
    for item in value:
        if isinstance(item, str):
            flags.add(item.strip().upper())
    return flags


def filter_history_for_scientific_eval(
    history_df: pd.DataFrame,
    min_quality_score: float,
    exclude_quality_flags: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove low-quality sensor rows before scientific evaluation.

    Filtering is important because sensor anomalies (tracking errors, buffer
    overflows) produce occupancy values that are physically impossible and
    uncorrelated with any learnable pattern.  Including them in evaluation would
    unfairly penalise the model for errors that no forecast could avoid.

    Two independent filter criteria are applied:
      1. Minimum quality score: rows with quality_score < min_quality_score removed.
      2. Hard quality flags: rows with any flag in exclude_quality_flags removed.

    Args:
        history_df: Raw occupancy history DataFrame with optional quality columns.
        min_quality_score: Rows below this threshold are excluded (0.6 = default).
        exclude_quality_flags: List of flag strings (e.g. ["TRACK_ERROR"]).

    Returns:
        Tuple of (filtered_df, stats_dict) where stats_dict contains row counts
        and filter parameters for the evaluation report.
    """
    if history_df.empty:
        return history_df.copy(), {
            "rows_before": 0,
            "rows_after": 0,
            "rows_dropped": 0,
            "min_quality_score": min_quality_score,
            "exclude_quality_flags": exclude_quality_flags,
        }

    hard_flags = {flag.strip().upper() for flag in exclude_quality_flags if flag.strip()}

    df = history_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "quality_score" not in df.columns:
        df["quality_score"] = 1.0
    if "quality_flags" not in df.columns:
        df["quality_flags"] = [[] for _ in range(len(df))]
    if "utilization" not in df.columns:
        df["utilization"] = 0.0

    before = len(df)

    quality_mask = pd.to_numeric(df["quality_score"], errors="coerce").fillna(0.0) >= float(min_quality_score)
    if hard_flags:
        flags_mask = ~df["quality_flags"].apply(lambda flags: bool(_quality_flag_set(flags) & hard_flags))
    else:
        flags_mask = pd.Series([True] * len(df), index=df.index)

    filtered = (
        df[quality_mask & flags_mask]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return filtered, {
        "rows_before": int(before),
        "rows_after": int(len(filtered)),
        "rows_dropped": int(before - len(filtered)),
        "min_quality_score": float(min_quality_score),
        "exclude_quality_flags": sorted(hard_flags),
    }


def _history_to_series(history_df: pd.DataFrame) -> pd.Series:
    if history_df.empty:
        return pd.Series(dtype=float)

    df = history_df[["timestamp", "occupancy"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    series = pd.to_numeric(df["occupancy"], errors="coerce").astype(float)
    return series.resample("1min").mean().interpolate(limit_direction="both")


def _effective_step_minutes(horizon: int) -> int:
    if horizon > 24 * 60:
        return 60
    return 1


def _baseline_point_forecast(series: pd.Series, horizon: int, capacity: float = 100.0) -> dict[str, float]:
    """Compute a blended seasonal + linear regression baseline forecast.

    This is a reference forecast that the trained model should outperform.
    It blends:
      - Seasonal component (60%): the value from `seasonal_lag` steps ago,
        representing 'same time last hour'.
      - Linear regression component (40%): a linear model on [time, sin(tod), cos(tod)]
        fitted on the history series.  This captures linear trends.

    The blend is clipped to [0, capacity] so the baseline obeys physical limits.
    Prediction intervals are estimated from historical residuals (seasonal diffs).

    Note: This function is also used by the live inference service (_forecast_baseline
    in main.py) which applies the same clipping after the April 2026 capacity fix.

    Args:
        series: UTC-indexed 1-min occupancy series (recent history).
        horizon: Forecast horizon in minutes.
        capacity: Upper bound for clipping predictions.

    Returns:
        Dict with 'yhat' (point forecast), 'q10' and 'q90' (interval bounds).
    """
    if series.empty:
        return {"yhat": 0.0, "q10": -2.0, "q90": 2.0}

    step_minutes = _effective_step_minutes(horizon)
    steps = max(1, math.ceil(horizon / step_minutes))

    values = series.values.astype(float)
    seasonal_lag = max(1, int(round(60 / step_minutes)))

    seasonal_base: list[float] = []
    for i in range(1, steps + 1):
        idx = len(values) - seasonal_lag + i - 1
        if 0 <= idx < len(values):
            seasonal_base.append(float(values[idx]))
        else:
            seasonal_base.append(float(values[-1]))

    n = len(values)
    x = np.arange(n)
    minute_of_day = np.array([ts.hour * 60 + ts.minute for ts in series.index])
    sin_t = np.sin(2 * np.pi * minute_of_day / 1440)
    cos_t = np.cos(2 * np.pi * minute_of_day / 1440)
    X = np.column_stack([x, sin_t, cos_t])

    # Closed-form linear regression to avoid extra estimator state in evaluation loop.
    beta, *_ = np.linalg.lstsq(X, values, rcond=None)

    future_x = np.arange(n, n + steps)
    last_ts = series.index[-1]
    future_idx = [last_ts + timedelta(minutes=step_minutes * i) for i in range(1, steps + 1)]
    future_minute = np.array([ts.hour * 60 + ts.minute for ts in future_idx])
    future_X = np.column_stack(
        [
            future_x,
            np.sin(2 * np.pi * future_minute / 1440),
            np.cos(2 * np.pi * future_minute / 1440),
        ]
    )

    regression_pred = future_X @ beta
    yhat_path = np.clip(0.6 * np.array(seasonal_base) + 0.4 * regression_pred, 0.0, capacity)

    if len(values) > seasonal_lag + 10:
        residuals = values[seasonal_lag:] - values[:-seasonal_lag]
    else:
        residuals = np.diff(values) if len(values) > 2 else np.array([0.0])

    q10 = float(np.quantile(residuals, 0.10)) if residuals.size else -2.0
    q90 = float(np.quantile(residuals, 0.90)) if residuals.size else 2.0
    yhat = float(yhat_path[-1])

    return {
        "yhat": yhat,
        "q10": float(yhat + q10),
        "q90": float(yhat + q90),
    }


def _normal_cdf(x: float) -> float:
    """Compute the standard normal CDF using the error function.

    Used by the Diebold-Mariano test to convert a Z-score to a p-value.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def diebold_mariano_test(
    model_abs_errors: list[float],
    baseline_abs_errors: list[float],
    horizon_steps: int,
) -> dict[str, Any]:
    """Perform the Diebold-Mariano (DM) test for forecast accuracy comparison.

    The DM test is a formal statistical hypothesis test that answers:
    "Is the model's forecast significantly more accurate than the baseline?"

    Null hypothesis H0: Model and baseline have equal forecast accuracy.
    Alternative H1 (one-sided): Model has lower absolute error (better accuracy).

    The test statistic is:
        DM = d_mean / sqrt(var(d) / n)
    where d_i = |error_model_i| - |error_baseline_i| (loss differential).

    A Newey-West HAC variance estimator is used to account for serial correlation
    in forecast errors (errors at nearby time steps are often correlated), which
    is the key innovation of the DM test over a simple t-test.

    Interpretation:
        p_one_sided_model_better < 0.05 → model is significantly better (95% conf.)
        stat < 0 → model has lower mean error (positive direction for improvement)

    Args:
        model_abs_errors: List of absolute errors from the new model.
        baseline_abs_errors: List of absolute errors from the baseline.
        horizon_steps: Forecast horizon in steps (used to set HAC lag truncation).

    Returns:
        Dict with: n, stat (DM statistic), p_value_two_sided, p_value_one_sided_model_better,
        significant (bool: p_one_sided < 0.05 AND stat < 0).
    """
    if len(model_abs_errors) != len(baseline_abs_errors):
        n = min(len(model_abs_errors), len(baseline_abs_errors))
        model_abs_errors = model_abs_errors[:n]
        baseline_abs_errors = baseline_abs_errors[:n]

    n = len(model_abs_errors)
    if n < 10:
        return {
            "n": n,
            "stat": 0.0,
            "p_value_two_sided": 1.0,
            "p_value_one_sided_model_better": 1.0,
            "significant": False,
        }

    d = np.asarray(model_abs_errors, dtype=float) - np.asarray(baseline_abs_errors, dtype=float)
    d_mean = float(np.mean(d))
    centered = d - d_mean

    lag = int(max(1, min(horizon_steps, n - 1, 24)))
    gamma0 = float(np.dot(centered, centered) / n)
    var_hat = gamma0

    for k in range(1, lag + 1):
        gamma = float(np.dot(centered[k:], centered[:-k]) / n)
        weight = 1.0 - k / (lag + 1.0)
        var_hat += 2.0 * weight * gamma

    var_d = max(var_hat / n, 1e-12)
    stat = d_mean / math.sqrt(var_d)
    p_two_sided = max(0.0, min(1.0, 2.0 * (1.0 - _normal_cdf(abs(stat)))))
    p_one_sided_model_better = max(0.0, min(1.0, _normal_cdf(stat)))

    return {
        "n": n,
        "stat": float(stat),
        "p_value_two_sided": float(p_two_sided),
        "p_value_one_sided_model_better": float(p_one_sided_model_better),
        "significant": bool(stat < 0.0 and p_one_sided_model_better < 0.05),
    }


def _pinball(y_true: np.ndarray, q_pred: np.ndarray, alpha: float) -> float:
    """Compute the pinball (quantile) loss for a single quantile level alpha."""
    delta = y_true - q_pred
    loss = np.where(delta >= 0, alpha * delta, (alpha - 1.0) * delta)
    return float(np.mean(loss))


def _naive_scale(series: pd.Series) -> float:
    """Compute the naive scale for MASE (Mean Absolute Scaled Error).

    MASE normalises MAE by the mean absolute first-order difference of the
    training series.  A MASE < 1 means the model beats the naive 1-step-ahead
    persistence forecast.  Returns 1.0 as a safe fallback if scale is degenerate.
    """
    values = series.values.astype(float)
    if values.size < 2:
        return 1.0
    diff = np.abs(np.diff(values))
    scale = float(np.mean(diff))
    if not math.isfinite(scale) or scale <= 1e-6:
        return 1.0
    return scale


def _compute_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a list of prediction samples into evaluation metrics.

    Each sample is a dict with keys: y_true, yhat, q10, q50, q90, mase_scale.
    Computes MAE, RMSE, MASE, pinball loss (q10/q50/q90 average), and coverage.
    """
    if not samples:
        return {
            "n": 0,
            "mae": None,
            "rmse": None,
            "mase": None,
            "pinball": None,
            "coverage90": None,
        }

    y = np.array([float(item["y_true"]) for item in samples], dtype=float)
    yhat = np.array([float(item["yhat"]) for item in samples], dtype=float)
    q10 = np.array([float(item["q10"]) for item in samples], dtype=float)
    q50 = np.array([float(item["q50"]) for item in samples], dtype=float)
    q90 = np.array([float(item["q90"]) for item in samples], dtype=float)
    scales = np.array([max(1e-6, float(item.get("mase_scale", 1.0))) for item in samples], dtype=float)

    abs_err = np.abs(y - yhat)
    sq_err = np.square(y - yhat)

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(sq_err)))
    mase = float(np.mean(abs_err / scales))

    p10 = _pinball(y, q10, 0.1)
    p50 = _pinball(y, q50, 0.5)
    p90 = _pinball(y, q90, 0.9)
    pinball = float((p10 + p50 + p90) / 3.0)

    coverage = float(np.mean((y >= q10) & (y <= q90)))

    return {
        "n": int(len(samples)),
        "mae": mae,
        "rmse": rmse,
        "mase": mase,
        "pinball": pinball,
        "coverage90": coverage,
        "pinball_q10": p10,
        "pinball_q50": p50,
        "pinball_q90": p90,
    }


def _unevaluated_metrics(sample_count: int) -> dict[str, Any]:
    return {
        "n": int(sample_count),
        "mae": None,
        "rmse": None,
        "mase": None,
        "pinball": None,
        "coverage90": None,
        "pinball_q10": None,
        "pinball_q50": None,
        "pinball_q90": None,
    }


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int, np.floating, float)):
        return bool(float(value))
    return False


def _build_segment_context(
    index: pd.DatetimeIndex,
    lecture_df: pd.DataFrame,
) -> pd.DataFrame:
    context = pd.DataFrame(
        {
            SEGMENT_FIELDS[SEGMENT_LECTURE_ACTIVE]: np.zeros(len(index), dtype=bool),
            SEGMENT_FIELDS[SEGMENT_HEAVY_EFFECT]: np.zeros(len(index), dtype=bool),
            SEGMENT_FIELDS[SEGMENT_LECTURE_TRANSITION_START]: np.zeros(len(index), dtype=bool),
            SEGMENT_FIELDS[SEGMENT_LECTURE_TRANSITION_END]: np.zeros(len(index), dtype=bool),
        },
        index=index,
    )
    if lecture_df.empty:
        return context

    local = lecture_df.copy()
    if "timestamp" not in local.columns:
        return context

    local["timestamp"] = pd.to_datetime(local["timestamp"], utc=True)
    local = local.sort_values("timestamp").set_index("timestamp")
    if "metadata" not in local.columns:
        local["metadata"] = [{} for _ in range(len(local))]

    def _meta_float(metadata: Any, key: str) -> float:
        if not isinstance(metadata, dict):
            return 0.0
        try:
            value = float(metadata.get(key, 0.0))
        except Exception:
            return 0.0
        if not math.isfinite(value):
            return 0.0
        return value

    for col in ("active_lectures", "starts_next_60m", "ends_next_60m"):
        if col not in local.columns:
            local[col] = 0.0

    local["heavy_active_lectures"] = local["metadata"].apply(lambda item: _meta_float(item, "heavy_active_lectures"))
    local["heavy_ended_last_60m"] = local["metadata"].apply(lambda item: _meta_float(item, "heavy_ended_last_60m"))

    minute = (
        local[
            [
                "active_lectures",
                "starts_next_60m",
                "ends_next_60m",
                "heavy_active_lectures",
                "heavy_ended_last_60m",
            ]
        ]
        .resample("1min")
        .mean(numeric_only=True)
        .reindex(index)
    )

    for col in minute.columns:
        minute[col] = minute[col].interpolate(limit_direction="both").fillna(0.0)

    context[SEGMENT_FIELDS[SEGMENT_LECTURE_ACTIVE]] = minute["active_lectures"] > 0.0
    context[SEGMENT_FIELDS[SEGMENT_HEAVY_EFFECT]] = (
        minute["heavy_active_lectures"] + minute["heavy_ended_last_60m"]
    ) > 0.0
    context[SEGMENT_FIELDS[SEGMENT_LECTURE_TRANSITION_START]] = minute["starts_next_60m"] > 0.0
    context[SEGMENT_FIELDS[SEGMENT_LECTURE_TRANSITION_END]] = minute["ends_next_60m"] > 0.0
    return context


def _sample_segment_flags(
    segment_context: pd.DataFrame,
    origin: pd.Timestamp,
) -> dict[str, bool]:
    if origin in segment_context.index:
        row = segment_context.loc[origin]
        return {column: _coerce_bool(row[column]) for column in SEGMENT_FIELDS.values()}
    return {column: False for column in SEGMENT_FIELDS.values()}


def _guarded_metrics(
    samples: list[dict[str, Any]],
    min_required_samples: int,
) -> tuple[dict[str, Any], bool]:
    if len(samples) < int(max(1, min_required_samples)):
        return _unevaluated_metrics(len(samples)), True
    return _compute_metrics(samples), False


def _compute_segment_metrics(
    samples: list[dict[str, Any]],
    min_required_samples: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for segment_name, flag_key in SEGMENT_FIELDS.items():
        val_samples = [s for s in samples if s.get("split") == "val" and _coerce_bool(s.get(flag_key, False))]
        test_samples = [s for s in samples if s.get("split") == "test" and _coerce_bool(s.get(flag_key, False))]
        val_metrics, val_insufficient = _guarded_metrics(val_samples, min_required_samples=min_required_samples)
        test_metrics, test_insufficient = _guarded_metrics(test_samples, min_required_samples=min_required_samples)

        payload[segment_name] = {
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "val_samples": int(len(val_samples)),
            "test_samples": int(len(test_samples)),
            "val_insufficient_samples": bool(val_insufficient),
            "test_insufficient_samples": bool(test_insufficient),
            "min_required_samples": int(max(1, min_required_samples)),
        }
    return payload


def _lecture_impact_summary(lecture_df: pd.DataFrame, enabled: bool) -> dict[str, Any]:
    empty_payload = {
        "enabled": enabled,
        "rows": 0,
        "minutes_with_heavy_effect_share": 0.0,
        "feature_availability_rate": 0.0,
        "lecture_net_pull": {"mean": None, "q10": None, "q50": None, "q90": None},
        "impact_model_versions": [],
    }
    if not enabled:
        return empty_payload

    if lecture_df.empty:
        return empty_payload

    metadata_series = (
        lecture_df["metadata"]
        if "metadata" in lecture_df.columns
        else pd.Series([{} for _ in range(len(lecture_df))], index=lecture_df.index)
    )

    def _meta_float(metadata: Any, key: str) -> float:
        if not isinstance(metadata, dict):
            return float("nan")
        try:
            out = float(metadata.get(key))
        except Exception:
            return float("nan")
        if not math.isfinite(out):
            return float("nan")
        return out

    heavy_now = metadata_series.apply(lambda value: _meta_float(value, "heavy_active_lectures"))
    heavy_post = metadata_series.apply(lambda value: _meta_float(value, "heavy_ended_last_60m"))
    net_pull = metadata_series.apply(lambda value: _meta_float(value, "lecture_net_pull"))
    versions = sorted(
        {
            str(value.get("impact_model_version")).strip()
            for value in metadata_series
            if isinstance(value, dict) and str(value.get("impact_model_version", "")).strip()
        }
    )

    availability = net_pull.notna()
    availability_rate = float(availability.mean()) if len(availability) else 0.0
    heavy_effect = (heavy_now.fillna(0.0) + heavy_post.fillna(0.0)) > 0.0
    heavy_effect_share = float(heavy_effect.mean()) if len(heavy_effect) else 0.0

    net_pull_valid = net_pull[net_pull.notna()]
    if len(net_pull_valid):
        net_pull_summary = {
            "mean": float(net_pull_valid.mean()),
            "q10": float(net_pull_valid.quantile(0.10)),
            "q50": float(net_pull_valid.quantile(0.50)),
            "q90": float(net_pull_valid.quantile(0.90)),
        }
    else:
        net_pull_summary = {"mean": None, "q10": None, "q50": None, "q90": None}

    return {
        "enabled": True,
        "rows": int(len(lecture_df)),
        "minutes_with_heavy_effect_share": heavy_effect_share,
        "feature_availability_rate": availability_rate,
        "lecture_net_pull": net_pull_summary,
        "impact_model_versions": versions,
    }


def _build_fold_boundaries(index: pd.DatetimeIndex, config: EvaluationConfig) -> list[dict[str, Any]]:
    if len(index) < 10:
        return []

    train_min = int(config.train_days * 24 * 60)
    val_min = int(config.val_days * 24 * 60)
    test_min = int(config.test_days * 24 * 60)
    gap = int(config.gap_minutes)

    fold_size = train_min + gap + val_min + gap + test_min
    if fold_size <= 0:
        return []

    bounds: list[dict[str, Any]] = []
    n = len(index)
    for fold in range(config.folds):
        test_end = n - fold * test_min
        test_start = test_end - test_min
        val_end = test_start - gap
        val_start = val_end - val_min
        train_end = val_start - gap
        train_start = train_end - train_min

        if train_start < 0:
            break

        bounds.append(
            {
                "fold_index": fold,
                "train_start": index[train_start],
                "train_end": index[train_end - 1],
                "val_start": index[val_start],
                "val_end": index[val_end - 1],
                "test_start": index[test_start],
                "test_end": index[test_end - 1],
                "train_rows": train_end - train_start,
                "val_rows": val_end - val_start,
                "test_rows": test_end - test_start,
            }
        )

    bounds.reverse()
    for i, item in enumerate(bounds):
        item["fold_number"] = i + 1
    return bounds


def _slice_frame(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, ts_col: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    local = df.copy()
    local[ts_col] = pd.to_datetime(local[ts_col], utc=True)
    mask = (local[ts_col] >= start) & (local[ts_col] <= end)
    return local.loc[mask].reset_index(drop=True)


def _slice_events(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    local = df.copy()
    local["starts_at"] = pd.to_datetime(local["starts_at"], utc=True)
    local["ends_at"] = pd.to_datetime(local["ends_at"], utc=True)
    mask = (local["ends_at"] >= start) & (local["starts_at"] <= end)
    return local.loc[mask].reset_index(drop=True)


def _select_origins(
    index: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    horizon: int,
    stride: int,
    max_origins: int,
) -> list[pd.Timestamp]:
    if len(index) == 0:
        return []

    latest_origin = end - timedelta(minutes=horizon)
    if latest_origin < start:
        return []

    candidates = index[(index >= start) & (index <= latest_origin)]
    if len(candidates) == 0:
        return []

    step = max(1, int(stride))
    candidates = candidates[::step]

    if len(candidates) > max_origins:
        sample_idx = np.linspace(0, len(candidates) - 1, max_origins).astype(int)
        candidates = candidates[sample_idx]

    return [pd.Timestamp(ts).tz_convert("UTC") for ts in candidates]


def _evaluate_baseline(
    series: pd.Series,
    origins: list[pd.Timestamp],
    horizon: int,
    fold_number: int,
    split: str,
    mase_scale: float,
    segment_context: pd.DataFrame,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for origin in origins:
        target_ts = origin + timedelta(minutes=horizon)
        if target_ts not in series.index:
            continue

        history = series.loc[:origin]
        forecast = _baseline_point_forecast(history, horizon=horizon)
        y_true = float(series.loc[target_ts])
        q10 = float(min(forecast["q10"], forecast["q90"]))
        q90 = float(max(forecast["q10"], forecast["q90"]))
        yhat = float(max(0.0, forecast["yhat"]))  # already clipped in _baseline_point_forecast

        samples.append(
            {
                "origin": origin.isoformat(),
                "target_ts": target_ts.isoformat(),
                "fold": int(fold_number),
                "split": split,
                "y_true": y_true,
                "yhat": yhat,
                "q10": q10,
                "q50": yhat,
                "q90": q90,
                "mase_scale": mase_scale,
                **_sample_segment_flags(segment_context=segment_context, origin=origin),
            }
        )
    return samples


def _train_gbdt_models(X: np.ndarray, y: np.ndarray, seed: int) -> dict[str, Any]:
    common = {
        "n_estimators": 240,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_samples_leaf": 4,
        "subsample": 0.9,
        "random_state": seed,
    }

    point = GradientBoostingRegressor(loss="squared_error", **common)
    q10 = GradientBoostingRegressor(loss="quantile", alpha=0.10, **common)
    q50 = GradientBoostingRegressor(loss="quantile", alpha=0.50, **common)
    q90 = GradientBoostingRegressor(loss="quantile", alpha=0.90, **common)

    point.fit(X, y)
    q10.fit(X, y)
    q50.fit(X, y)
    q90.fit(X, y)

    return {
        "point": point,
        "q10": q10,
        "q50": q50,
        "q90": q90,
    }


def _feature_row_for_origin(
    history_df: pd.DataFrame,
    events_df: pd.DataFrame,
    lecture_df: pd.DataFrame,
    origin: pd.Timestamp,
    horizon: int,
    include_lecture_impact: bool,
) -> pd.DataFrame | None:
    hist_slice = _slice_frame(
        df=history_df,
        start=pd.Timestamp("1970-01-01", tz="UTC"),
        end=origin,
        ts_col="timestamp",
    )
    if hist_slice.empty:
        return None

    events_slice = _slice_events(
        df=events_df,
        start=hist_slice["timestamp"].min(),
        end=origin + timedelta(minutes=horizon),
    )
    lecture_slice = _slice_frame(
        df=lecture_df,
        start=hist_slice["timestamp"].min(),
        end=origin + timedelta(minutes=horizon),
        ts_col="timestamp",
    )

    feature_df = build_feature_frame(
        history_df=hist_slice,
        events_df=events_slice,
        lecture_df=lecture_slice,
        use_calendar_features=True,
        include_lecture_impact=include_lecture_impact,
    )
    if feature_df.empty:
        return None
    row = feature_df.tail(1)
    if row.empty:
        return None
    return row


def _evaluate_gbdt_fold(
    history_df: pd.DataFrame,
    events_df: pd.DataFrame,
    lecture_df: pd.DataFrame,
    segment_context: pd.DataFrame,
    series: pd.Series,
    fold: dict[str, Any],
    horizon: int,
    origins: dict[str, list[pd.Timestamp]],
    seed: int,
    include_lecture_impact: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    return [], ["tf_mlp_removed_lgbm_only"]

    warnings: list[str] = []

    train_history = _slice_frame(history_df, fold["train_start"], fold["train_end"], "timestamp")
    if len(train_history) < 600:
        return [], [f"insufficient_train_history_fold_{fold['fold_number']}"]

    train_events = _slice_events(events_df, fold["train_start"], fold["train_end"] + timedelta(minutes=horizon))
    train_lecture = _slice_frame(
        lecture_df,
        fold["train_start"],
        fold["train_end"] + timedelta(minutes=horizon),
        "timestamp",
    )

    train_features = build_feature_frame(
        history_df=train_history,
        events_df=train_events,
        lecture_df=train_lecture,
        use_calendar_features=True,
        include_lecture_impact=include_lecture_impact,
    )
    if train_features.empty:
        return [], [f"feature_frame_empty_fold_{fold['fold_number']}"]

    train_set = train_features.copy()
    train_set["target"] = train_set["occupancy"].shift(-horizon)
    train_set = train_set.dropna()
    if len(train_set) < 400:
        return [], [f"insufficient_supervised_rows_fold_{fold['fold_number']}"]

    feature_cols = [c for c in train_set.columns if c not in {"target"}]
    X_train = train_set[feature_cols].values.astype(np.float32)
    y_train = train_set["target"].values.astype(np.float32)

    models = _train_gbdt_models(X_train, y_train, seed)

    mase_scale = _naive_scale(_history_to_series(train_history))
    samples: list[dict[str, Any]] = []

    for split_name, split_origins in origins.items():
        for origin in split_origins:
            target_ts = origin + timedelta(minutes=horizon)
            if target_ts not in series.index:
                continue
            row = _feature_row_for_origin(
                history_df=history_df,
                events_df=events_df,
                lecture_df=lecture_df,
                origin=origin,
                horizon=horizon,
                include_lecture_impact=include_lecture_impact,
            )
            if row is None:
                continue

            row_X = row[feature_cols].values.astype(np.float32)
            y_true = float(series.loc[target_ts])
            yhat = float(max(0.0, models["point"].predict(row_X)[0]))
            q10 = float(models["q10"].predict(row_X)[0])
            q50 = float(models["q50"].predict(row_X)[0])
            q90 = float(models["q90"].predict(row_X)[0])
            low = float(min(q10, q90))
            high = float(max(q10, q90))

            samples.append(
                {
                    "origin": origin.isoformat(),
                    "target_ts": target_ts.isoformat(),
                    "fold": int(fold["fold_number"]),
                    "split": split_name,
                    "y_true": y_true,
                    "yhat": yhat,
                    "q10": low,
                    "q50": q50,
                    "q90": high,
                    "mase_scale": mase_scale,
                    **_sample_segment_flags(segment_context=segment_context, origin=origin),
                }
            )

    return samples, warnings


def _evaluate_tf_fold(
    history_df: pd.DataFrame,
    events_df: pd.DataFrame,
    lecture_df: pd.DataFrame,
    segment_context: pd.DataFrame,
    series: pd.Series,
    fold: dict[str, Any],
    horizon: int,
    origins: dict[str, list[pd.Timestamp]],
    zone_id: str,
    config: EvaluationConfig,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []

    if horizon > config.tf_max_horizon_minutes:
        return [], [f"horizon_{horizon}_exceeds_tf_eval_limit_{config.tf_max_horizon_minutes}"]

    train_history = _slice_frame(history_df, fold["train_start"], fold["train_end"], "timestamp")
    if len(train_history) < max(config.tf_min_train_points, 300):
        return [], [f"insufficient_tf_history_fold_{fold['fold_number']}"]

    train_events = _slice_events(events_df, fold["train_start"], fold["train_end"] + timedelta(minutes=horizon))
    train_lecture = _slice_frame(
        lecture_df,
        fold["train_start"],
        fold["train_end"] + timedelta(minutes=horizon),
        "timestamp",
    )

    min_points = min(
        max(config.tf_min_train_points, 300),
        max(300, len(train_history) - 30),
    )

    with tempfile.TemporaryDirectory(prefix="sitcheck-eval-tf-") as tmpdir:
        try:
            result = train_zone_model(
                history_df=train_history,
                events_df=train_events,
                lecture_df=train_lecture,
                config=TrainingConfig(
                    model_dir=tmpdir,
                    zone_id=zone_id,
                    horizon=horizon,
                    product="short_term",
                    min_train_points=min_points,
                    use_calendar_features=True,
                    include_lecture_impact=config.include_lecture_impact,
                    min_quality_score=0.0,
                    epochs=config.tf_epochs,
                    batch_size=config.tf_batch_size,
                    verbose=0,
                    random_seed=config.random_seed,
                ),
            )
        except Exception as exc:
            return [], [f"tf_train_error_fold_{fold['fold_number']}: {exc}"]

        bundle = load_bundle(tmpdir, zone_id, horizon)
        feature_cols = bundle["metadata"].get("feature_columns", [])
        residuals = bundle.get("residuals", {})
        q10_arr = np.array(residuals.get("q10", []), dtype=float)
        q90_arr = np.array(residuals.get("q90", []), dtype=float)
        q10_res = float(q10_arr[horizon - 1]) if q10_arr.size >= horizon else float(q10_arr[-1] if q10_arr.size else -2.0)
        q90_res = float(q90_arr[horizon - 1]) if q90_arr.size >= horizon else float(q90_arr[-1] if q90_arr.size else 2.0)

        mase_scale = _naive_scale(_history_to_series(train_history))
        samples: list[dict[str, Any]] = []

        for split_name, split_origins in origins.items():
            for origin in split_origins:
                target_ts = origin + timedelta(minutes=horizon)
                if target_ts not in series.index:
                    continue
                row = _feature_row_for_origin(
                    history_df=history_df,
                    events_df=events_df,
                    lecture_df=lecture_df,
                    origin=origin,
                    horizon=horizon,
                    include_lecture_impact=config.include_lecture_impact,
                )
                if row is None:
                    continue

                try:
                    vector, _ = build_inference_vector(row, feature_cols)
                except Exception:
                    continue

                X = vector.values.astype(np.float32)
                X_scaled = bundle["scaler"].transform(X).astype(np.float32)
                pred_vec = bundle["model"].predict(X_scaled, verbose=0)[0]
                yhat = float(max(0.0, float(pred_vec[horizon - 1])))
                q10 = float(yhat + q10_res)
                q90 = float(yhat + q90_res)

                samples.append(
                    {
                        "origin": origin.isoformat(),
                        "target_ts": target_ts.isoformat(),
                        "fold": int(fold["fold_number"]),
                        "split": split_name,
                        "y_true": float(series.loc[target_ts]),
                        "yhat": yhat,
                        "q10": float(min(q10, q90)),
                        "q50": yhat,
                        "q90": float(max(q10, q90)),
                        "mase_scale": mase_scale,
                        "fold_model_version": result.get("model_version"),
                        **_sample_segment_flags(segment_context=segment_context, origin=origin),
                    }
                )

        return samples, warnings


def _evaluate_sarimax(
    series: pd.Series,
    origins: list[pd.Timestamp],
    horizon: int,
    fold_number: int,
    split: str,
    mase_scale: float,
    segment_context: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []

    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception as exc:
        return [], [f"statsmodels_unavailable: {exc}"]

    samples: list[dict[str, Any]] = []
    if not origins:
        return samples, warnings

    max_origins = 20
    if len(origins) > max_origins:
        idx = np.linspace(0, len(origins) - 1, max_origins).astype(int)
        selected_origins = [origins[i] for i in idx]
    else:
        selected_origins = origins

    for origin in selected_origins:
        target_ts = origin + timedelta(minutes=horizon)
        if target_ts not in series.index:
            continue

        series_to_origin = series.loc[:origin]
        if len(series_to_origin) < 200:
            continue

        try:
            model = SARIMAX(
                series_to_origin.values,
                order=(2, 1, 2),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False)
            forecast = fitted.get_forecast(steps=horizon)
            mean_pred = np.asarray(forecast.predicted_mean, dtype=float)
            conf = forecast.conf_int(alpha=0.10)

            yhat = float(max(0.0, mean_pred[-1]))
            q10 = float(conf[-1][0]) if np.ndim(conf) == 2 else float(yhat - 2.0)
            q90 = float(conf[-1][1]) if np.ndim(conf) == 2 else float(yhat + 2.0)
            y_true = float(series.loc[target_ts])

            samples.append(
                {
                    "origin": origin.isoformat(),
                    "target_ts": target_ts.isoformat(),
                    "fold": int(fold_number),
                    "split": split,
                    "y_true": y_true,
                    "yhat": yhat,
                    "q10": float(min(q10, q90)),
                    "q50": yhat,
                    "q90": float(max(q10, q90)),
                    "mase_scale": mase_scale,
                    **_sample_segment_flags(segment_context=segment_context, origin=origin),
                }
            )
        except Exception as exc:
            warnings.append(f"sarimax_forecast_error_fold_{fold_number}: {exc}")

    return samples, warnings


def _split_samples(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    val = [s for s in samples if s.get("split") == "val"]
    test = [s for s in samples if s.get("split") == "test"]
    return val, test


def _composite_score(
    model_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> float | None:
    if model_metrics.get("mae") is None or baseline_metrics.get("mae") is None:
        return None

    base_mae = max(float(baseline_metrics["mae"]), 1e-6)
    base_mase = max(float(baseline_metrics.get("mase") or 1e-6), 1e-6)
    base_pinball = max(float(baseline_metrics.get("pinball") or 1e-6), 1e-6)

    mae_norm = float(model_metrics["mae"]) / base_mae
    mase_norm = float(model_metrics.get("mase") or 0.0) / base_mase
    pinball_norm = float(model_metrics.get("pinball") or 0.0) / base_pinball

    coverage = float(model_metrics.get("coverage90") or 0.0)
    coverage_penalty = min(1.0, abs(coverage - 0.9) / 0.1)

    return float(0.50 * mae_norm + 0.20 * mase_norm + 0.20 * pinball_norm + 0.10 * coverage_penalty)


def evaluate_training_run(
    history_df: pd.DataFrame,
    events_df: pd.DataFrame,
    lecture_df: pd.DataFrame,
    config: EvaluationConfig,
) -> dict[str, Any]:
    """Run a full scientific evaluation of all forecast models on historical data.

    This is the top-level evaluation entry point.  It orchestrates:
      1. Quality filtering of the input history.
      2. For each fold: train models, run rolling-origin evaluation, collect samples.
      3. Aggregate samples across folds into metrics per horizon per model.
      4. Run Diebold-Mariano tests for statistical significance.
      5. Perform segment analysis (lecture-active, heavy, transitions).
      6. Apply the promotion gate: decide if the new model should be deployed.
      7. Build and return the full evaluation report dict.

    The promotion gate checks:
      - model MAE < baseline MAE × (1 - improvement_threshold)  [e.g. 8% better]
      - Coverage in [coverage_low, coverage_high]               [e.g. 85–95%]
      - No long-horizon degradation beyond max_long_degradation  [e.g. 2%]
      - DM test significant (p < 0.05) for primary horizon

    Args:
        history_df: Full occupancy history DataFrame with timestamps and quality cols.
        events_df: Calendar events DataFrame.
        lecture_df: DHBW lecture activity DataFrame.
        config: EvaluationConfig with all evaluation parameters.

    Returns:
        Comprehensive evaluation report dict ready for save_evaluation_report().

    Raises:
        ValueError: If no history remains after quality filtering.
    """
    filtered_history, quality_info = filter_history_for_scientific_eval(
        history_df=history_df,
        min_quality_score=config.min_quality_score,
        exclude_quality_flags=config.exclude_quality_flags,
    )

    if filtered_history.empty:
        raise ValueError("no history left after quality filtering")

    series = _history_to_series(filtered_history)
    if series.empty:
        raise ValueError("history series is empty after resampling")

    segment_context = _build_segment_context(index=series.index, lecture_df=lecture_df)

    folds = _build_fold_boundaries(series.index, config)
    if len(folds) < config.folds:
        raise ValueError(
            f"insufficient history for {config.folds} folds (available={len(folds)}). "
            f"Increase history or reduce folds/train/val/test windows."
        )

    run_id = f"eval-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"

    model_samples: dict[str, dict[int, list[dict[str, Any]]]] = {
        MODEL_BASELINE: {},
        MODEL_TF: {},
        MODEL_GBDT: {},
        MODEL_SARIMAX: {},
    }
    model_warnings: dict[str, list[str]] = {
        MODEL_BASELINE: [],
        MODEL_TF: [],
        MODEL_GBDT: [],
        MODEL_SARIMAX: [],
    }

    for horizon in config.horizons:
        model_samples[MODEL_BASELINE][horizon] = []
        model_samples[MODEL_TF][horizon] = []
        model_samples[MODEL_GBDT][horizon] = []
        model_samples[MODEL_SARIMAX][horizon] = []

        for fold in folds:
            val_origins = _select_origins(
                index=series.index,
                start=fold["val_start"],
                end=fold["val_end"],
                horizon=horizon,
                stride=config.origin_stride_minutes,
                max_origins=config.max_origins_per_split,
            )
            test_origins = _select_origins(
                index=series.index,
                start=fold["test_start"],
                end=fold["test_end"],
                horizon=horizon,
                stride=config.origin_stride_minutes,
                max_origins=config.max_origins_per_split,
            )

            if not val_origins and not test_origins:
                model_warnings[MODEL_BASELINE].append(
                    f"no_origins_fold_{fold['fold_number']}_h{horizon}"
                )
                continue

            train_series = series.loc[fold["train_start"]:fold["train_end"]]
            mase_scale = _naive_scale(train_series)

            model_samples[MODEL_BASELINE][horizon].extend(
                _evaluate_baseline(
                    series=series,
                    origins=val_origins,
                    horizon=horizon,
                    fold_number=fold["fold_number"],
                    split="val",
                    mase_scale=mase_scale,
                    segment_context=segment_context,
                )
            )
            model_samples[MODEL_BASELINE][horizon].extend(
                _evaluate_baseline(
                    series=series,
                    origins=test_origins,
                    horizon=horizon,
                    fold_number=fold["fold_number"],
                    split="test",
                    mase_scale=mase_scale,
                    segment_context=segment_context,
                )
            )

            origins = {
                "val": val_origins,
                "test": test_origins,
            }

            gbdt_samples, gbdt_warnings = _evaluate_gbdt_fold(
                history_df=filtered_history,
                events_df=events_df,
                lecture_df=lecture_df,
                segment_context=segment_context,
                series=series,
                fold=fold,
                horizon=horizon,
                origins=origins,
                seed=config.random_seed,
                include_lecture_impact=config.include_lecture_impact,
            )
            model_samples[MODEL_GBDT][horizon].extend(gbdt_samples)
            model_warnings[MODEL_GBDT].extend(gbdt_warnings)

            if config.enable_tf:
                tf_samples, tf_warnings = _evaluate_tf_fold(
                    history_df=filtered_history,
                    events_df=events_df,
                    lecture_df=lecture_df,
                    segment_context=segment_context,
                    series=series,
                    fold=fold,
                    horizon=horizon,
                    origins=origins,
                    zone_id=config.zone_id,
                    config=config,
                )
                model_samples[MODEL_TF][horizon].extend(tf_samples)
                model_warnings[MODEL_TF].extend(tf_warnings)

            if config.enable_sarimax:
                sarimax_val, sarimax_val_warnings = _evaluate_sarimax(
                    series=series,
                    origins=val_origins,
                    horizon=horizon,
                    fold_number=fold["fold_number"],
                    split="val",
                    mase_scale=mase_scale,
                    segment_context=segment_context,
                )
                sarimax_test, sarimax_test_warnings = _evaluate_sarimax(
                    series=series,
                    origins=test_origins,
                    horizon=horizon,
                    fold_number=fold["fold_number"],
                    split="test",
                    mase_scale=mase_scale,
                    segment_context=segment_context,
                )
                model_samples[MODEL_SARIMAX][horizon].extend(sarimax_val)
                model_samples[MODEL_SARIMAX][horizon].extend(sarimax_test)
                model_warnings[MODEL_SARIMAX].extend(sarimax_val_warnings)
                model_warnings[MODEL_SARIMAX].extend(sarimax_test_warnings)

    report_models: dict[str, Any] = {}
    for model_name, per_horizon in model_samples.items():
        horizon_payload: dict[str, Any] = {}
        for horizon, samples in per_horizon.items():
            val_samples, test_samples = _split_samples(samples)
            horizon_payload[str(horizon)] = {
                "val_metrics": _compute_metrics(val_samples),
                "test_metrics": _compute_metrics(test_samples),
                "val_samples": int(len(val_samples)),
                "test_samples": int(len(test_samples)),
                "segments": _compute_segment_metrics(
                    samples=samples,
                    min_required_samples=config.segment_min_samples,
                ),
            }
        report_models[model_name] = {
            "horizons": horizon_payload,
            "warnings": sorted(set(model_warnings.get(model_name, []))),
        }

    primary_horizon = int(config.primary_horizon)
    primary_key = str(primary_horizon)
    baseline_primary_metrics = report_models[MODEL_BASELINE]["horizons"].get(primary_key, {}).get("test_metrics", {})

    comparison: dict[str, Any] = {
        "primary_horizon": primary_horizon,
        "composite_scores": {},
        "dm_test": {},
        "improvement_vs_baseline_mae": {},
        "long_horizon_degradation": {},
    }

    candidates = [MODEL_GBDT]
    for model_name in candidates:
        model_primary_metrics = report_models[model_name]["horizons"].get(primary_key, {}).get("test_metrics", {})
        score = _composite_score(model_primary_metrics, baseline_primary_metrics)
        comparison["composite_scores"][model_name] = score

        mae_model = model_primary_metrics.get("mae")
        mae_base = baseline_primary_metrics.get("mae")
        if mae_model is not None and mae_base not in (None, 0):
            improvement = float((float(mae_base) - float(mae_model)) / max(float(mae_base), 1e-6))
            comparison["improvement_vs_baseline_mae"][model_name] = improvement
        else:
            comparison["improvement_vs_baseline_mae"][model_name] = None

        base_samples = [
            s for s in model_samples[MODEL_BASELINE].get(primary_horizon, []) if s.get("split") == "test"
        ]
        model_test_samples = [
            s for s in model_samples[model_name].get(primary_horizon, []) if s.get("split") == "test"
        ]

        base_map = {(s["fold"], s["target_ts"]): s for s in base_samples}
        model_map = {(s["fold"], s["target_ts"]): s for s in model_test_samples}
        shared_keys = sorted(set(base_map).intersection(model_map))

        model_abs = [abs(float(model_map[key]["y_true"]) - float(model_map[key]["yhat"])) for key in shared_keys]
        base_abs = [abs(float(base_map[key]["y_true"]) - float(base_map[key]["yhat"])) for key in shared_keys]
        dm = diebold_mariano_test(
            model_abs_errors=model_abs,
            baseline_abs_errors=base_abs,
            horizon_steps=max(1, primary_horizon // max(1, config.origin_stride_minutes)),
        )
        comparison["dm_test"][model_name] = dm

        long_deg: dict[str, Any] = {}
        for horizon in sorted(set(config.horizons) - {primary_horizon}):
            h_key = str(horizon)
            model_h = report_models[model_name]["horizons"].get(h_key, {}).get("test_metrics", {})
            base_h = report_models[MODEL_BASELINE]["horizons"].get(h_key, {}).get("test_metrics", {})
            mae_h_model = model_h.get("mae")
            mae_h_base = base_h.get("mae")
            if mae_h_model is None or mae_h_base in (None, 0):
                long_deg[h_key] = None
            else:
                long_deg[h_key] = float((float(mae_h_model) - float(mae_h_base)) / max(float(mae_h_base), 1e-6))
        comparison["long_horizon_degradation"][model_name] = long_deg

    def _score_or_inf(value: float | None) -> float:
        if value is None or not math.isfinite(value):
            return float("inf")
        return float(value)

    champion_model = min(candidates, key=lambda name: _score_or_inf(comparison["composite_scores"].get(name)))
    champion_score = comparison["composite_scores"].get(champion_model)
    champion_improvement = comparison["improvement_vs_baseline_mae"].get(champion_model)
    champion_dm = comparison["dm_test"].get(champion_model, {})
    champion_long = comparison["long_horizon_degradation"].get(champion_model, {})

    champion_metrics = report_models[champion_model]["horizons"].get(primary_key, {}).get("test_metrics", {})
    coverage = champion_metrics.get("coverage90")

    decision_reasons: list[str] = []

    if champion_score is None:
        decision_reasons.append("champion_missing_primary_metrics")

    pass_improvement = champion_improvement is not None and champion_improvement >= config.improvement_threshold
    if not pass_improvement:
        decision_reasons.append(
            f"improvement_below_threshold({champion_improvement} < {config.improvement_threshold})"
        )

    pass_significance = bool(champion_dm.get("significant", False))
    if not pass_significance:
        decision_reasons.append("dm_not_significant")

    pass_coverage = (
        coverage is not None and config.coverage_low <= float(coverage) <= config.coverage_high
    )
    if not pass_coverage:
        decision_reasons.append(
            f"coverage_out_of_bounds({coverage} not in [{config.coverage_low},{config.coverage_high}])"
        )

    missing_long = False
    pass_long = True
    for horizon in sorted(set(config.horizons) - {primary_horizon}):
        key = str(horizon)
        value = champion_long.get(key)
        if value is None:
            missing_long = True
            pass_long = False
            decision_reasons.append(f"missing_long_horizon_metric_h{horizon}")
            continue
        if float(value) > config.max_long_degradation:
            pass_long = False
            decision_reasons.append(
                f"long_horizon_degradation_h{horizon}_too_high({value} > {config.max_long_degradation})"
            )

    scientific_pass = bool(pass_improvement and pass_significance and pass_coverage and pass_long)

    promotable_backend = bool(scientific_pass and champion_model == MODEL_GBDT and not missing_long)
    if scientific_pass and champion_model != MODEL_GBDT:
        decision_reasons.append("champion_not_runtime_deployable_current_backend")

    promotion_horizons: list[int] = []
    if promotable_backend:
        promotion_horizons = [primary_horizon]
        for horizon in sorted(set(config.horizons) - {primary_horizon}):
            if champion_long.get(str(horizon)) is not None:
                promotion_horizons.append(horizon)

    report = {
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "zone_id": config.zone_id,
        "config": {
            "horizons": config.horizons,
            "primary_horizon": config.primary_horizon,
            "folds": config.folds,
            "train_days": config.train_days,
            "val_days": config.val_days,
            "test_days": config.test_days,
            "gap_minutes": config.gap_minutes,
            "origin_stride_minutes": config.origin_stride_minutes,
            "max_origins_per_split": config.max_origins_per_split,
            "min_quality_score": config.min_quality_score,
            "exclude_quality_flags": config.exclude_quality_flags,
            "improvement_threshold": config.improvement_threshold,
            "max_long_degradation": config.max_long_degradation,
            "coverage_low": config.coverage_low,
            "coverage_high": config.coverage_high,
            "enable_tf": config.enable_tf,
            "enable_sarimax": config.enable_sarimax,
            "include_lecture_impact": config.include_lecture_impact,
            "segment_min_samples": int(max(1, config.segment_min_samples)),
        },
        "data_quality": quality_info,
        "lecture_impact_summary": _lecture_impact_summary(
            lecture_df=lecture_df,
            enabled=config.include_lecture_impact,
        ),
        "folds": [
            {
                "fold_number": item["fold_number"],
                "train_start": item["train_start"].isoformat(),
                "train_end": item["train_end"].isoformat(),
                "val_start": item["val_start"].isoformat(),
                "val_end": item["val_end"].isoformat(),
                "test_start": item["test_start"].isoformat(),
                "test_end": item["test_end"].isoformat(),
                "train_rows": item["train_rows"],
                "val_rows": item["val_rows"],
                "test_rows": item["test_rows"],
            }
            for item in folds
        ],
        "models": report_models,
        "comparison": comparison,
        "decision": {
            "scientific_pass": scientific_pass,
            "champion_model": champion_model,
            "champion_score": champion_score,
            "promotable_backend": promotable_backend,
            "promotion_horizons": promotion_horizons,
            "reasons": sorted(set(decision_reasons)) if decision_reasons else ["OK"],
        },
        "hypothesis_test": {
            "h0": "candidate not better than baseline",
            "h1": "candidate better than baseline with statistical significance",
            "method": "rolling-origin backtesting + Diebold-Mariano one-sided test",
        },
    }

    return report
