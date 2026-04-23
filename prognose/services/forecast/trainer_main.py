"""Nightly model trainer daemon for the Sitcheck forecast service.

This script runs as a long-lived background process (Docker service
'forecast-trainer', port 8013).  It periodically triggers model retraining
for all configured zones by calling the forecast service REST API.

Design rationale:
  - Training is decoupled from inference: the forecast service handles HTTP
    requests, while this daemon handles nightly batch training.
  - Training is triggered via REST (POST /v1/train/batch) rather than direct
    function calls, so the trainer can run in a separate container with no
    access to the database or model files directly.
  - The trainer operates in a simple polling loop: train all zones, sleep,
    repeat.  No scheduler framework is used to keep dependencies minimal.

Important: The forecast service must be in FORECAST_TRAINING_MODE=maintenance
for the /v1/train/batch endpoint to accept requests.  If it is in the default
'locked' mode, all training calls will return HTTP 423 and be silently skipped.

Key symbols:
    train_once: Send a training request for one zone.
    main: Entry point; runs the infinite polling loop.
"""
from __future__ import annotations

import os
import time
from datetime import UTC, datetime

import requests


# ---------------------------------------------------------------------------
# Configuration (environment variables with defaults)
# ---------------------------------------------------------------------------

# URL of the forecast microservice.  In Docker Compose this resolves via the
# service name 'forecast' on the internal Docker network.
FORECAST_SERVICE_URL = os.getenv("FORECAST_SERVICE_URL", "http://forecast:8001").rstrip("/")

# Comma-separated list of zone IDs to train.  Defaults to the single default zone.
TRAIN_ZONES = [zone.strip() for zone in os.getenv("TF_TRAIN_ZONES", os.getenv("DEFAULT_ZONE_ID", "default-zone")).split(",") if zone.strip()]

# Single-horizon fallback (used if the batch endpoint is unavailable).
TRAIN_HORIZON = int(os.getenv("TF_DEFAULT_HORIZON", "210"))

# List of horizons to train in each batch run (default: H210 and H1440).
TRAIN_HORIZONS = [int(h.strip()) for h in os.getenv("TF_TRAIN_HORIZONS", "210,1440").split(",") if h.strip()]

# How many minutes to sleep between training iterations.  Default: 6 hours.
RETRAIN_INTERVAL_MINUTES = int(os.getenv("TF_RETRAIN_INTERVAL_MINUTES", "360"))

# How many hours of occupancy history to send to the training pipeline.
TRAIN_HISTORY_HOURS = int(os.getenv("TF_TRAIN_HISTORY_HOURS", str(24 * 30)))


def train_once(zone_id: str) -> None:
    """Trigger one training run for `zone_id` via the forecast service API.

    Tries the batch endpoint first (trains all horizons in one call).
    Falls back to the single-horizon endpoint for backward compatibility with
    older versions of the forecast service that may not have /v1/train/batch.

    Args:
        zone_id: Zone identifier to train (e.g. "default-zone").
    """
    batch_payload = {
        "zone_id": zone_id,
        "horizons": TRAIN_HORIZONS,
        "history_hours": TRAIN_HISTORY_HOURS,
        "full_retrain": False,
    }
    # Timeout of 1800s (30 min) to allow for long neural-network training runs.
    batch_resp = requests.post(f"{FORECAST_SERVICE_URL}/v1/train/batch", json=batch_payload, timeout=1800)
    if batch_resp.ok:
        data = batch_resp.json()
        print(
            f"[trainer] zone={zone_id} batch status={data.get('status')} "
            f"summary={data.get('summary')}"
        )
        return

    # Backward-compatibility fallback if batch endpoint is unavailable.
    payload = {
        "zone_id": zone_id,
        "horizon": TRAIN_HORIZON,
        "history_hours": TRAIN_HISTORY_HOURS,
        "full_retrain": False,
    }
    resp = requests.post(f"{FORECAST_SERVICE_URL}/v1/train", json=payload, timeout=1800)
    if not resp.ok:
        print(f"[trainer] zone={zone_id} train failed: {resp.status_code} {resp.text}")
        return
    data = resp.json()
    print(f"[trainer] zone={zone_id} trained model={data.get('model_version')} metrics={data.get('metrics')}")


def main() -> None:
    """Run the trainer daemon: train all zones, sleep, repeat.

    The loop runs indefinitely.  Each iteration trains every zone in TRAIN_ZONES
    sequentially, then sleeps for RETRAIN_INTERVAL_MINUTES before the next cycle.

    Exceptions from individual zones are caught and logged so that a failure
    in one zone does not prevent training for the remaining zones.
    """
    print(
        f"[trainer] start at {datetime.now(UTC).isoformat()} "
        f"zones={TRAIN_ZONES} horizons={TRAIN_HORIZONS} interval_min={RETRAIN_INTERVAL_MINUTES}"
    )
    while True:
        for zone in TRAIN_ZONES:
            try:
                train_once(zone)
            except Exception as exc:  # pragma: no cover
                print(f"[trainer] zone={zone} unexpected failure: {exc}")
        # Sleep between cycles.  time.sleep() is used instead of a scheduler
        # to keep the daemon simple and avoid additional dependencies.
        time.sleep(max(60, RETRAIN_INTERVAL_MINUTES * 60))


if __name__ == "__main__":
    main()
