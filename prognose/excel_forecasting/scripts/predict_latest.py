from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_and_prepare_data
from src.torch_model import TorchForecaster
from src.windows import inverse_transform_y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate latest multi-horizon forecast from trained artifacts.")
    parser.add_argument("--file", required=True, help="Path to Excel file")
    parser.add_argument("--artifacts_dir", default="artifacts", help="Training artifacts directory")
    parser.add_argument("--model", choices=["tf", "torch"], default="tf", help="Model backend")
    parser.add_argument("--sheet", default=0, help="Excel sheet name or index")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output JSON path. Default: <artifacts_dir>/live/forecast_latest_<model>.json",
    )
    return parser.parse_args()


def _load_config(artifacts_dir: Path) -> dict:
    config_path = artifacts_dir / "common" / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json: {config_path}. Train a model first.")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_output_path(artifacts_dir: Path, model_name: str, output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg).expanduser().resolve()
    live_dir = artifacts_dir / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    return (live_dir / f"forecast_latest_{model_name}.json").resolve()


def _get_effective_freq(prepared_index: pd.DatetimeIndex, configured_freq: str | None) -> pd.tseries.offsets.BaseOffset:
    if configured_freq:
        return pd.tseries.frequencies.to_offset(configured_freq)

    if len(prepared_index) < 2:
        raise ValueError("Cannot infer frequency from fewer than 2 timestamps.")

    delta = prepared_index.to_series().diff().dropna().median()
    if pd.isna(delta) or delta <= pd.Timedelta(0):
        raise ValueError("Unable to infer a valid frequency from data.")

    return pd.tseries.frequencies.to_offset(delta)


def _predict_tf(X_last_scaled: np.ndarray, artifacts_dir: Path) -> np.ndarray:
    model_path = artifacts_dir / "tf" / "model_best.keras"
    scaler_path = artifacts_dir / "tf" / "y_scaler.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing TF model artifact: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing TF y_scaler artifact: {scaler_path}")

    model = tf.keras.models.load_model(model_path)
    y_scaler = joblib.load(scaler_path)

    pred_scaled = model.predict(X_last_scaled, verbose=0)
    pred = inverse_transform_y(pred_scaled, y_scaler)
    return pred.astype(np.float32).reshape(-1)


def _predict_torch(X_last_scaled: np.ndarray, artifacts_dir: Path, feature_count: int, horizon: int) -> np.ndarray:
    model_path = artifacts_dir / "torch" / "model_best.pt"
    scaler_path = artifacts_dir / "torch" / "y_scaler.pkl"
    meta_path = artifacts_dir / "torch" / "meta.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing Torch model artifact: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing Torch y_scaler artifact: {scaler_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing Torch meta artifact: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    params = meta.get("params", {})
    model = TorchForecaster(
        input_size=feature_count,
        horizon=horizon,
        rnn_type=meta.get("rnn_type", "lstm"),
        hidden_size=int(params.get("hidden_size", 64)),
        num_layers=int(params.get("num_layers", 1)),
        dropout=float(params.get("dropout", 0.1)),
    )

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    y_scaler = joblib.load(scaler_path)

    with torch.no_grad():
        xb = torch.from_numpy(X_last_scaled.astype(np.float32))
        pred_scaled = model(xb).numpy()

    pred = inverse_transform_y(pred_scaled, y_scaler)
    return pred.astype(np.float32).reshape(-1)


def main() -> None:
    args = parse_args()

    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
    config = _load_config(artifacts_dir)

    lookback = int(config["lookback"])
    horizon = int(config["horizon"])
    date_col = str(config["date_col"])
    target_col = str(config["target_col"])
    feature_cols = list(config["feature_cols"])
    configured_freq = config.get("freq")

    prepared = load_and_prepare_data(
        file_path=args.file,
        sheet=args.sheet,
        date_col=date_col,
        target_col=target_col,
        freq=configured_freq,
        features=None,
        seasonal_period_override=config.get("seasonal_period"),
    )

    missing = [c for c in feature_cols if c not in prepared.frame.columns]
    if missing:
        raise ValueError(
            f"Input data does not provide required trained features: {missing}. "
            "Retrain model on this data schema or provide matching columns."
        )

    if len(prepared.frame) < lookback:
        raise ValueError(
            f"Not enough rows for inference. Need at least lookback={lookback}, got {len(prepared.frame)}"
        )

    X_raw = prepared.frame[feature_cols].values.astype(np.float32)

    x_scaler_path = artifacts_dir / args.model / "x_scaler.pkl"
    if not x_scaler_path.exists():
        raise FileNotFoundError(f"Missing x_scaler artifact: {x_scaler_path}")
    x_scaler = joblib.load(x_scaler_path)

    X_scaled = x_scaler.transform(X_raw)
    X_last = X_scaled[-lookback:, :].reshape(1, lookback, len(feature_cols)).astype(np.float32)

    if args.model == "tf":
        yhat = _predict_tf(X_last, artifacts_dir)
        model_version = "tf_latest"
    else:
        yhat = _predict_torch(X_last, artifacts_dir, feature_count=len(feature_cols), horizon=horizon)
        model_version = "torch_latest"

    last_ts = pd.Timestamp(prepared.frame.index[-1])
    if last_ts.tzinfo is not None:
        last_ts = last_ts.tz_convert("UTC").tz_localize(None)

    offset = _get_effective_freq(prepared.frame.index, configured_freq)
    future_ts = [last_ts + (i + 1) * offset for i in range(horizon)]

    result = {
        "model": args.model,
        "model_version": model_version,
        "source_file": str(Path(args.file).expanduser().resolve()),
        "target_col": target_col,
        "lookback": lookback,
        "horizon": horizon,
        "last_timestamp": last_ts.isoformat(),
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "forecast": [
            {
                "step": i + 1,
                "timestamp": pd.Timestamp(ts).isoformat(),
                "yhat": float(yhat[i]),
            }
            for i, ts in enumerate(future_ts)
        ],
    }

    out_path = _resolve_output_path(artifacts_dir, args.model, args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"Wrote forecast: {out_path}")
    print(f"last_timestamp={result['last_timestamp']} horizon={horizon} model={args.model}")


if __name__ == "__main__":
    main()
