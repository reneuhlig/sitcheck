#!/usr/bin/env python3
"""Train LGBM quantile models for every 15-minute horizon from h15 to h180.

Produces 12 independent models so the 3-hour forecast trajectory can use
dedicated LGBM predictions at each step instead of interpolation.

Horizon mapping (horizon_steps × 15 min):
  steps=1  → h15    steps=5  → h75    steps=9  → h135
  steps=2  → h30    steps=6  → h90    steps=10 → h150
  steps=3  → h45    steps=7  → h105   steps=11 → h165
  steps=4  → h60    steps=8  → h120   steps=12 → h180

Usage (run inside the forecast container):
  python train_multihorizon.py --data /app/training_data.parquet
  python train_multihorizon.py --data /app/training_data.parquet --skip-existing
  python train_multihorizon.py --data /app/training_data.parquet --horizons 30,45,75
  python train_multihorizon.py --data /app/training_data.parquet --dry-run

The script follows the same pipeline as train_gbdt.py:
  1. Load the Parquet training data.
  2. Walk-forward cross-validation (6 folds by default).
  3. Promotion gate: >= 8% improvement over persistence baseline, 85-98% coverage.
  4. Train final model on 85/15 chronological split.
  5. Save bundle to model_dir/{zone_id}/h{horizon_minutes}/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Allow imports from this directory when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_gbdt import GBDTConfig
from model_store import gbdt_model_status
from train_gbdt import TrainGBDTConfig, run_full_pipeline

# ---------------------------------------------------------------------------
# Horizon definitions
# ---------------------------------------------------------------------------

# All 15-minute step horizons covering 3 hours.
ALL_3H_HORIZONS_MIN = [steps * 15 for steps in range(1, 13)]  # [15,30,...,180]

# Default model directory (matches TF_MODEL_DIR env var default in docker-compose).
DEFAULT_MODEL_DIR = os.getenv("TF_MODEL_DIR", "/models")
DEFAULT_ZONE_ID = os.getenv("DEFAULT_ZONE_ID", "default-zone")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_horizons(raw: str) -> list[int]:
    """Parse a comma-separated list of horizon minutes."""
    result = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        h = int(token)
        if h <= 0 or h % 15 != 0:
            raise ValueError(f"horizon {h} is not a positive multiple of 15 minutes")
        result.append(h)
    if not result:
        raise ValueError("no valid horizons in list")
    return sorted(set(result))


def _horizon_promoted(model_dir: str, zone_id: str, horizon_min: int) -> bool:
    """Return True if a promoted LGBM model already exists for this horizon."""
    status = gbdt_model_status(model_dir, zone_id, horizon_min)
    return bool(status.get("exists") and status.get("promoted"))


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.1f}min"


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_horizons(
    data_path: str,
    model_dir: str,
    zone_id: str,
    horizons_min: list[int],
    n_folds: int,
    gbdt_cfg: GBDTConfig,
    skip_existing: bool,
    dry_run: bool,
) -> list[dict]:
    """Train (or skip) each horizon and return a list of result dicts."""
    results = []
    total = len(horizons_min)

    for idx, h_min in enumerate(horizons_min, start=1):
        steps = h_min // 15
        label = f"h{h_min:03d} (step {idx}/{total}, {steps}×15min)"

        if dry_run:
            status = "would_train"
            if skip_existing and _horizon_promoted(model_dir, zone_id, h_min):
                status = "would_skip"
            print(f"  [DRY-RUN] {label}: {status}")
            results.append({"horizon": h_min, "status": status})
            continue

        if skip_existing and _horizon_promoted(model_dir, zone_id, h_min):
            print(f"\n[{idx}/{total}] {label} → SKIP (promoted model already exists)")
            results.append({"horizon": h_min, "status": "skipped", "reason": "promoted_exists"})
            continue

        print(f"\n{'=' * 66}")
        print(f"[{idx}/{total}] Training {label}")
        print(f"{'=' * 66}")

        t0 = time.monotonic()
        try:
            config = TrainGBDTConfig(
                model_dir=model_dir,
                zone_id=zone_id,
                horizon_steps=steps,
                training_data_path=data_path,
                n_folds=n_folds,
                gbdt_config=gbdt_cfg,
            )
            result = run_full_pipeline(config)
            elapsed = time.monotonic() - t0

            promoted = result.get("promoted", False)
            mae = result.get("metrics", {}).get("weighted_mae", float("nan"))
            improvement = result.get("metrics", {}).get("improvement_vs_persistence_pct", float("nan"))
            coverage = result.get("metrics", {}).get("weighted_coverage", float("nan"))
            model_version = result.get("model_version", "?")

            status_label = "PROMOTED ✓" if promoted else "NOT PROMOTED ✗"
            print(f"\n  {status_label} | model={model_version}")
            print(f"  MAE={mae:.3f}  improvement={improvement:.1f}%  coverage={coverage:.1f}%")
            print(f"  Duration: {_format_duration(elapsed)}")

            results.append({
                "horizon": h_min,
                "status": "promoted" if promoted else "trained_not_promoted",
                "model_version": model_version,
                "mae": round(mae, 4),
                "improvement_pct": round(improvement, 2),
                "coverage_pct": round(coverage, 2),
                "duration_s": round(elapsed, 1),
            })

        except Exception as exc:
            elapsed = time.monotonic() - t0
            print(f"\n  ERROR after {_format_duration(elapsed)}: {exc}")
            results.append({
                "horizon": h_min,
                "status": "error",
                "error": str(exc),
                "duration_s": round(elapsed, 1),
            })

    return results


def _print_summary(results: list[dict], total_s: float) -> None:
    print(f"\n{'=' * 66}")
    print("SUMMARY")
    print(f"{'=' * 66}")
    print(f"{'Horizon':>8}  {'Status':<24}  {'MAE':>6}  {'Impr%':>6}  {'Cov%':>6}  {'Time':>7}")
    print("-" * 66)
    promoted_count = 0
    skipped_count = 0
    error_count = 0
    for r in results:
        h = r["horizon"]
        s = r["status"]
        mae = r.get("mae", "")
        impr = r.get("improvement_pct", "")
        cov = r.get("coverage_pct", "")
        dur = r.get("duration_s", "")
        mae_s = f"{mae:.3f}" if isinstance(mae, float) else str(mae)
        impr_s = f"{impr:.1f}" if isinstance(impr, float) else str(impr)
        cov_s = f"{cov:.1f}" if isinstance(cov, float) else str(cov)
        dur_s = f"{dur:.0f}s" if isinstance(dur, float) else str(dur)
        print(f"{h:>8}  {s:<24}  {mae_s:>6}  {impr_s:>6}  {cov_s:>6}  {dur_s:>7}")
        if s == "promoted":
            promoted_count += 1
        elif s == "skipped":
            skipped_count += 1
        elif s == "error":
            error_count += 1
    print("-" * 66)
    print(f"Total: {len(results)} horizons | promoted: {promoted_count} | "
          f"skipped: {skipped_count} | errors: {error_count}")
    print(f"Total time: {_format_duration(total_s)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train LGBM models for every 15-minute horizon from h15 to h180.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to training_data.parquet (e.g. /app/training_data.parquet)",
    )
    parser.add_argument(
        "--model-dir", default=DEFAULT_MODEL_DIR,
        help=f"Model output directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--zone-id", default=DEFAULT_ZONE_ID,
        help=f"Zone identifier (default: {DEFAULT_ZONE_ID})",
    )
    parser.add_argument(
        "--horizons",
        default=",".join(str(h) for h in ALL_3H_HORIZONS_MIN),
        help="Comma-separated list of horizon minutes to train "
             "(default: all 15-min steps 15..180)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip horizons that already have a promoted model on disk",
    )
    parser.add_argument(
        "--n-folds", type=int, default=6,
        help="Walk-forward CV folds (default: 6)",
    )
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be trained without executing training",
    )
    parser.add_argument(
        "--json-output", metavar="FILE",
        help="Write full results to this JSON file after training",
    )

    args = parser.parse_args()

    # Validate inputs
    data_path = Path(args.data)
    if not args.dry_run and not data_path.exists():
        print(f"ERROR: training data not found: {data_path}", file=sys.stderr)
        print("  Use --data to specify the path to training_data.parquet", file=sys.stderr)
        return 1

    horizons = _parse_horizons(args.horizons)
    gbdt_cfg = GBDTConfig(
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
    )

    print(f"Multi-horizon LGBM training")
    print(f"  data:        {data_path}")
    print(f"  model_dir:   {args.model_dir}")
    print(f"  zone_id:     {args.zone_id}")
    print(f"  horizons:    {horizons} ({len(horizons)} models)")
    print(f"  skip_exist:  {args.skip_existing}")
    print(f"  n_folds:     {args.n_folds}")
    print(f"  dry_run:     {args.dry_run}")

    t_start = time.monotonic()
    results = train_horizons(
        data_path=str(data_path),
        model_dir=args.model_dir,
        zone_id=args.zone_id,
        horizons_min=horizons,
        n_folds=args.n_folds,
        gbdt_cfg=gbdt_cfg,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
    )
    total_elapsed = time.monotonic() - t_start
    _print_summary(results, total_elapsed)

    if args.json_output:
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"results": results, "total_duration_s": round(total_elapsed, 1)}, indent=2),
            encoding="utf-8",
        )
        print(f"\nJSON results written to: {out_path}")

    errors = sum(1 for r in results if r["status"] == "error")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
