from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import compute_baselines
from src.data import load_and_prepare_data, parse_feature_list
from src.tf_model import TfTrainConfig, train_tf_model
from src.windows import export_common_artifacts, inverse_transform_y, prepare_windowed_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TensorFlow/Keras multi-horizon forecaster from Excel data.")
    parser.add_argument("--file", required=True, help="Path to Excel file")
    parser.add_argument("--sheet", default=0, help="Excel sheet name or index")
    parser.add_argument("--date_col", required=True, help="Datetime column name")
    parser.add_argument("--target_col", required=True, help="Target numeric column name")
    parser.add_argument("--freq", default=None, help="Optional resample frequency, e.g. 15min/H/D")
    parser.add_argument("--lookback", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--features", default=None, help="Comma-separated feature columns")
    parser.add_argument("--rnn_type", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--artifacts_dir", default="artifacts")
    parser.add_argument("--seasonal_period", type=int, default=None)
    parser.add_argument("--units", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    return parser.parse_args()


def _save_baselines(
    artifacts_dir: Path,
    target_test_series: np.ndarray,
    lookback: int,
    horizon: int,
    seasonal_period: int | None,
    expected_y_true: np.ndarray,
) -> None:
    baselines_dir = artifacts_dir / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)

    baseline_data = compute_baselines(
        target_test=target_test_series,
        lookback=lookback,
        horizon=horizon,
        seasonal_period=seasonal_period,
    )

    y_true = baseline_data["y_true"]
    naive = baseline_data["naive"]
    seasonal = baseline_data["seasonal_naive"]
    seasonal_reason = baseline_data["seasonal_reason"]

    if not isinstance(y_true, np.ndarray) or y_true.shape != expected_y_true.shape:
        raise RuntimeError(
            f"Baseline y_true shape mismatch: got {None if not isinstance(y_true, np.ndarray) else y_true.shape}, "
            f"expected {expected_y_true.shape}."
        )

    np.save(baselines_dir / "y_true.npy", y_true.astype(np.float32))
    np.save(baselines_dir / "pred_naive.npy", np.asarray(naive, dtype=np.float32))

    metadata = {
        "seasonal_period": seasonal_period,
        "seasonal_available": seasonal is not None,
        "seasonal_reason": seasonal_reason,
    }

    if isinstance(seasonal, np.ndarray):
        np.save(baselines_dir / "pred_seasonal_naive.npy", seasonal.astype(np.float32))

    with (baselines_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    args = parse_args()

    artifacts_root = Path(args.artifacts_dir)
    tf_dir = artifacts_root / "tf"
    tf_dir.mkdir(parents=True, exist_ok=True)

    features = parse_feature_list(args.features)

    prepared = load_and_prepare_data(
        file_path=args.file,
        sheet=args.sheet,
        date_col=args.date_col,
        target_col=args.target_col,
        freq=args.freq,
        features=features,
        seasonal_period_override=args.seasonal_period,
    )

    windowed = prepare_windowed_data(
        frame=prepared.frame,
        feature_cols=prepared.feature_cols,
        target_col=prepared.target_col,
        lookback=args.lookback,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    config = {
        "lookback": args.lookback,
        "horizon": args.horizon,
        "freq": prepared.inferred_freq,
        "date_col": args.date_col,
        "target_col": prepared.target_col,
        "feature_cols": prepared.feature_cols,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "seasonal_period": prepared.seasonal_period,
        "split_info": windowed.split_info,
    }
    export_common_artifacts(
        artifacts_dir=artifacts_root,
        y_test_raw=windowed.y_test_raw,
        timestamps_test=windowed.timestamps_test,
        y_scaler=windowed.y_scaler,
        config=config,
    )

    _save_baselines(
        artifacts_dir=artifacts_root,
        target_test_series=windowed.target_test_series,
        lookback=args.lookback,
        horizon=args.horizon,
        seasonal_period=prepared.seasonal_period,
        expected_y_true=windowed.y_test_raw,
    )

    train_cfg = TfTrainConfig(
        rnn_type=args.rnn_type,
        units=args.units,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
    )

    checkpoint_path = str(tf_dir / "checkpoint_best.keras")

    model, history = train_tf_model(
        X_train=windowed.X_train,
        y_train=windowed.y_train,
        X_val=windowed.X_val,
        y_val=windowed.y_val,
        config=train_cfg,
        checkpoint_path=checkpoint_path,
    )

    pred_test_scaled = model.predict(windowed.X_test, batch_size=args.batch_size, verbose=0)
    pred_test = inverse_transform_y(pred_test_scaled, windowed.y_scaler).astype(np.float32)

    if pred_test.shape != windowed.y_test_raw.shape:
        raise RuntimeError(
            f"TF prediction shape mismatch: got {pred_test.shape}, expected {windowed.y_test_raw.shape}."
        )

    model.save(tf_dir / "model_best.keras")
    joblib.dump(windowed.x_scaler, tf_dir / "x_scaler.pkl")
    joblib.dump(windowed.y_scaler, tf_dir / "y_scaler.pkl")
    np.save(tf_dir / "pred_test.npy", pred_test)

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(tf_dir / "history.csv", index=False)

    best_val = float(np.min(history.history.get("val_loss", [np.nan])))
    best_epoch = int(np.argmin(history.history.get("val_loss", [np.nan])) + 1)

    devices = [d.device_type for d in tf.config.list_physical_devices()]
    meta = {
        "framework": "tensorflow",
        "rnn_type": args.rnn_type,
        "devices": devices,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "params": {
            "lookback": args.lookback,
            "horizon": args.horizon,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "units": args.units,
            "dropout": args.dropout,
            "patience": args.patience,
            "seed": args.seed,
        },
        "split_info": windowed.split_info,
        "feature_count": len(prepared.feature_cols),
    }
    with (tf_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print(
        "TensorFlow training complete. "
        f"best_epoch={best_epoch}, best_val_loss={best_val:.6f}, "
        f"test_samples={windowed.split_info['test_samples']}"
    )
    print(f"Artifacts written to: {tf_dir}")


if __name__ == "__main__":
    main()
