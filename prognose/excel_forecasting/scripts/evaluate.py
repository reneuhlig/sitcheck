from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate forecasting artifacts and generate reports.")
    parser.add_argument("--artifacts_dir", default="artifacts")
    parser.add_argument("--example_idx", type=int, default=0)
    parser.add_argument("--rolling_max_points", type=int, default=300)
    return parser.parse_args()


def _load_required(path: Path, label: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact for {label}: {path}")
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Artifact {path} for {label} must be 2D, got shape {arr.shape}")
    return arr.astype(np.float32)


def _load_optional(path: Path, label: str, expected_shape: tuple[int, int]) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = np.load(path)
    if arr.shape != expected_shape:
        raise ValueError(
            f"Artifact shape mismatch for {label}: got {arr.shape}, expected {expected_shape}. Path: {path}"
        )
    return arr.astype(np.float32)


def _build_predictions_frame(
    y_true: np.ndarray,
    timestamps: pd.Series,
    preds: dict[str, np.ndarray | None],
) -> pd.DataFrame:
    n_samples, horizon = y_true.shape
    rows: list[dict[str, float | int | str | None]] = []

    for i in range(n_samples):
        anchor = str(timestamps.iloc[i]) if i < len(timestamps) else ""
        for step in range(horizon):
            row = {
                "sample_idx": i,
                "step": step + 1,
                "timestamp_anchor": anchor,
                "y_true": float(y_true[i, step]),
                "pred_torch": np.nan,
                "pred_tf": np.nan,
                "pred_naive": np.nan,
                "pred_seasonal_naive": np.nan,
            }
            if preds.get("torch") is not None:
                row["pred_torch"] = float(preds["torch"][i, step])
            if preds.get("tf") is not None:
                row["pred_tf"] = float(preds["tf"][i, step])
            if preds.get("naive") is not None:
                row["pred_naive"] = float(preds["naive"][i, step])
            if preds.get("seasonal_naive") is not None:
                row["pred_seasonal_naive"] = float(preds["seasonal_naive"][i, step])
            rows.append(row)

    return pd.DataFrame(rows)


def _plot_horizon_example(
    plot_path: Path,
    y_true: np.ndarray,
    preds: dict[str, np.ndarray | None],
    example_idx: int,
) -> None:
    n_samples, horizon = y_true.shape
    idx = min(max(example_idx, 0), max(n_samples - 1, 0))
    x = np.arange(1, horizon + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(x, y_true[idx], label="true", linewidth=2)

    for label, arr in preds.items():
        if arr is not None:
            plt.plot(x, arr[idx], label=label)

    plt.title(f"Horizon Forecast Example (sample_idx={idx})")
    plt.xlabel("Forecast step")
    plt.ylabel("Target")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def _plot_rolling_step1(
    plot_path: Path,
    y_true: np.ndarray,
    preds: dict[str, np.ndarray | None],
    max_points: int,
) -> None:
    n_samples = y_true.shape[0]
    if n_samples == 0:
        raise ValueError("No test samples available for rolling plot.")

    max_points = max(1, min(max_points, n_samples))
    start = n_samples - max_points
    x = np.arange(start, n_samples)

    plt.figure(figsize=(12, 5))
    plt.plot(x, y_true[start:, 0], label="true_step1", linewidth=2)

    for label, arr in preds.items():
        if arr is not None:
            plt.plot(x, arr[start:, 0], label=f"{label}_step1", alpha=0.9)

    plt.title("Rolling Forecast (Step=1)")
    plt.xlabel("Test sample index")
    plt.ylabel("Target")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    artifacts = Path(args.artifacts_dir)

    common_dir = artifacts / "common"
    plots_dir = artifacts / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    y_true = _load_required(common_dir / "y_test.npy", "common y_true")
    expected_shape = tuple(y_true.shape)

    ts_path = common_dir / "timestamps_test.csv"
    if not ts_path.exists():
        raise FileNotFoundError(f"Missing required timestamp artifact: {ts_path}")
    ts_df = pd.read_csv(ts_path)
    if "timestamp_anchor" not in ts_df.columns:
        raise ValueError(f"Missing 'timestamp_anchor' column in {ts_path}")

    preds: dict[str, np.ndarray | None] = {
        "torch": _load_optional(artifacts / "torch" / "pred_test.npy", "torch", expected_shape),
        "tf": _load_optional(artifacts / "tf" / "pred_test.npy", "tf", expected_shape),
        "naive": _load_optional(artifacts / "baselines" / "pred_naive.npy", "naive", expected_shape),
        "seasonal_naive": _load_optional(
            artifacts / "baselines" / "pred_seasonal_naive.npy", "seasonal_naive", expected_shape
        ),
    }

    if all(val is None for val in preds.values()):
        raise RuntimeError(
            f"No prediction artifacts found under {artifacts}. "
            "Run train_torch.py and/or train_tf.py first."
        )

    metrics: dict[str, dict[str, float] | dict[str, str]] = {}
    for name, arr in preds.items():
        if arr is not None:
            metrics[name] = compute_metrics(y_true, arr)

    baseline_meta_path = artifacts / "baselines" / "meta.json"
    if baseline_meta_path.exists():
        with baseline_meta_path.open("r", encoding="utf-8") as handle:
            baseline_meta = json.load(handle)
        if baseline_meta.get("seasonal_available") is False:
            metrics["seasonal_naive"] = {
                "status": "skipped",
                "reason": str(baseline_meta.get("seasonal_reason", "not available")),
            }

    with (artifacts / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    pred_frame = _build_predictions_frame(y_true=y_true, timestamps=ts_df["timestamp_anchor"], preds=preds)
    pred_frame.to_csv(artifacts / "predictions.csv", index=False)

    _plot_horizon_example(
        plot_path=plots_dir / "horizon_example.png",
        y_true=y_true,
        preds=preds,
        example_idx=args.example_idx,
    )
    _plot_rolling_step1(
        plot_path=plots_dir / "rolling_step1.png",
        y_true=y_true,
        preds=preds,
        max_points=args.rolling_max_points,
    )

    print(f"Evaluation complete. Metrics: {artifacts / 'metrics.json'}")
    print(f"Predictions table: {artifacts / 'predictions.csv'}")
    print(f"Plots: {plots_dir}")


if __name__ == "__main__":
    main()
