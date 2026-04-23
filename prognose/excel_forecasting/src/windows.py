from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class WindowedData:
    """Windowed and scaled splits for sequence forecasting."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    y_test_raw: np.ndarray
    target_test_series: np.ndarray
    timestamps_test: np.ndarray
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    split_info: dict[str, int]


def inverse_transform_y(y_scaled: np.ndarray, y_scaler: StandardScaler) -> np.ndarray:
    """Inverse-transform scaled multi-horizon targets."""
    if y_scaled.ndim == 1:
        y_2d = y_scaled.reshape(-1, 1)
        inv = y_scaler.inverse_transform(y_2d)
        return inv.ravel()

    flat = y_scaled.reshape(-1, 1)
    inv = y_scaler.inverse_transform(flat)
    return inv.reshape(y_scaled.shape)


def _split_frame(frame: pd.DataFrame, train_ratio: float, val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if val_ratio <= 0 or val_ratio >= 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1.")

    n = len(frame)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = frame.iloc[:train_end].copy()
    val_df = frame.iloc[train_end:val_end].copy()
    test_df = frame.iloc[val_end:].copy()

    return train_df, val_df, test_df


def _make_windows(
    features: np.ndarray,
    target: np.ndarray,
    index: pd.Index,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    ts_list: list[np.datetime64] = []

    end = len(features) - horizon + 1
    for i in range(lookback, end):
        X_list.append(features[i - lookback : i, :])
        y_list.append(target[i : i + horizon])
        anchor_ts = pd.Timestamp(index[i])
        if anchor_ts.tzinfo is not None:
            anchor_ts = anchor_ts.tz_convert("UTC").tz_localize(None)
        ts_list.append(np.datetime64(anchor_ts))

    if not X_list:
        return (
            np.empty((0, lookback, features.shape[1]), dtype=np.float32),
            np.empty((0, horizon), dtype=np.float32),
            np.empty((0,), dtype="datetime64[ns]"),
        )

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    timestamps = np.asarray(ts_list)
    return X, y, timestamps


def _make_target_windows(target: np.ndarray, lookback: int, horizon: int) -> np.ndarray:
    out: list[np.ndarray] = []
    end = len(target) - horizon + 1
    for i in range(lookback, end):
        out.append(target[i : i + horizon])

    if not out:
        return np.empty((0, horizon), dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def prepare_windowed_data(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    lookback: int,
    horizon: int,
    train_ratio: float,
    val_ratio: float,
) -> WindowedData:
    """Split chronologically, scale on train only, and create sliding windows."""
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if len(frame) < (lookback + horizon + 10):
        raise ValueError(
            f"Not enough rows ({len(frame)}) for lookback={lookback} and horizon={horizon}."
        )

    train_df, val_df, test_df = _split_frame(frame, train_ratio=train_ratio, val_ratio=val_ratio)

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if len(df) < (lookback + horizon):
            raise ValueError(
                f"Split '{name}' has too few rows ({len(df)}). "
                f"Need at least lookback+horizon={lookback + horizon}. "
                "Adjust ratios or reduce lookback/horizon."
            )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaler.fit(train_df[feature_cols].values)
    y_scaler.fit(train_df[[target_col]].values)

    train_x_scaled = x_scaler.transform(train_df[feature_cols].values)
    val_x_scaled = x_scaler.transform(val_df[feature_cols].values)
    test_x_scaled = x_scaler.transform(test_df[feature_cols].values)

    train_y_scaled = y_scaler.transform(train_df[[target_col]].values).ravel()
    val_y_scaled = y_scaler.transform(val_df[[target_col]].values).ravel()
    test_y_scaled = y_scaler.transform(test_df[[target_col]].values).ravel()

    X_train, y_train, _ = _make_windows(
        train_x_scaled,
        train_y_scaled,
        train_df.index,
        lookback=lookback,
        horizon=horizon,
    )
    X_val, y_val, _ = _make_windows(
        val_x_scaled,
        val_y_scaled,
        val_df.index,
        lookback=lookback,
        horizon=horizon,
    )
    X_test, y_test, ts_test = _make_windows(
        test_x_scaled,
        test_y_scaled,
        test_df.index,
        lookback=lookback,
        horizon=horizon,
    )

    y_test_raw = _make_target_windows(test_df[target_col].values.astype(np.float32), lookback=lookback, horizon=horizon)

    if y_test.shape != y_test_raw.shape:
        raise RuntimeError(
            f"Internal mismatch: y_test shape {y_test.shape} != y_test_raw shape {y_test_raw.shape}"
        )

    split_info = {
        "total_rows": int(len(frame)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
    }

    return WindowedData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        y_test_raw=y_test_raw,
        target_test_series=test_df[target_col].values.astype(np.float32),
        timestamps_test=ts_test,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        split_info=split_info,
    )


def export_common_artifacts(
    artifacts_dir: str | Path,
    y_test_raw: np.ndarray,
    timestamps_test: np.ndarray,
    y_scaler: StandardScaler,
    config: dict,
) -> None:
    """Write common test targets/config/scaler used by evaluation."""
    root = Path(artifacts_dir)
    common_dir = root / "common"
    common_dir.mkdir(parents=True, exist_ok=True)

    np.save(common_dir / "y_test.npy", y_test_raw)

    ts_df = pd.DataFrame({"timestamp_anchor": pd.to_datetime(timestamps_test)})
    ts_df.to_csv(common_dir / "timestamps_test.csv", index=False)

    with (common_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    joblib.dump(y_scaler, common_dir / "target_scaler.pkl")
