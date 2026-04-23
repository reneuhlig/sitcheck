from __future__ import annotations

import numpy as np


def _target_windows(target: np.ndarray, lookback: int, horizon: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    end = len(target) - horizon + 1
    for i in range(lookback, end):
        rows.append(target[i : i + horizon])
    if not rows:
        return np.empty((0, horizon), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def naive_forecast(target: np.ndarray, lookback: int, horizon: int) -> np.ndarray:
    """Naive baseline: repeat last observed value from lookback window."""
    preds: list[np.ndarray] = []
    end = len(target) - horizon + 1
    for i in range(lookback, end):
        last_val = float(target[i - 1])
        preds.append(np.full((horizon,), last_val, dtype=np.float32))

    if not preds:
        return np.empty((0, horizon), dtype=np.float32)
    return np.asarray(preds, dtype=np.float32)


def seasonal_naive_forecast(
    target: np.ndarray,
    lookback: int,
    horizon: int,
    seasonal_period: int,
) -> np.ndarray:
    """Seasonal naive baseline without leakage (recursive for long horizons)."""
    preds: list[np.ndarray] = []
    end = len(target) - horizon + 1

    for i in range(lookback, end):
        history = list(map(float, target[:i]))
        sample_pred: list[float] = []

        for _ in range(horizon):
            src_idx = len(history) - seasonal_period
            if src_idx < 0:
                pred_val = history[-1]
            else:
                pred_val = history[src_idx]
            sample_pred.append(float(pred_val))
            history.append(float(pred_val))

        preds.append(np.asarray(sample_pred, dtype=np.float32))

    if not preds:
        return np.empty((0, horizon), dtype=np.float32)
    return np.asarray(preds, dtype=np.float32)


def compute_baselines(
    target_test: np.ndarray,
    lookback: int,
    horizon: int,
    seasonal_period: int | None,
) -> dict[str, np.ndarray | str | None]:
    """Compute baseline predictions and return with reason on seasonal fallback."""
    y_true = _target_windows(target_test.astype(np.float32), lookback=lookback, horizon=horizon)
    naive_pred = naive_forecast(target_test.astype(np.float32), lookback=lookback, horizon=horizon)

    seasonal_pred: np.ndarray | None = None
    seasonal_reason: str | None = None

    if seasonal_period is None:
        seasonal_reason = "seasonal_period is None"
    elif seasonal_period <= 1:
        seasonal_reason = f"invalid seasonal_period={seasonal_period}"
    elif lookback < seasonal_period:
        seasonal_reason = (
            f"lookback={lookback} is smaller than seasonal_period={seasonal_period}"
        )
    else:
        seasonal_pred = seasonal_naive_forecast(
            target=target_test.astype(np.float32),
            lookback=lookback,
            horizon=horizon,
            seasonal_period=seasonal_period,
        )

    return {
        "y_true": y_true,
        "naive": naive_pred,
        "seasonal_naive": seasonal_pred,
        "seasonal_reason": seasonal_reason,
    }
