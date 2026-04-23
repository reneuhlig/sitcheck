#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import math
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient


def tf_available() -> bool:
    try:
        import tensorflow  # noqa: F401

        return True
    except Exception:
        return False


if not tf_available():
    print("forecast tf service smoke: tensorflow unavailable, skipping")
    raise SystemExit(0)

ROOT = Path(__file__).resolve().parents[2]
DB_FILE = ROOT / "phase3_tf_service_test.db"
MODEL_DIR = ROOT / ".tmp_models_tf_smoke"

if DB_FILE.exists():
    DB_FILE.unlink()
if MODEL_DIR.exists():
    import shutil

    shutil.rmtree(MODEL_DIR)

os.environ["DATABASE_URL"] = f"sqlite:///{DB_FILE}"
os.environ["FORECAST_MODEL_BACKEND"] = "tf_mlp"
os.environ["TF_MODEL_DIR"] = str(MODEL_DIR)
os.environ["TF_MIN_TRAIN_POINTS"] = "200"
os.environ["TF_TRAIN_EPOCHS"] = "3"
os.environ["TF_BATCH_SIZE"] = "64"
os.environ["TF_TRAIN_HISTORY_HOURS"] = "72"
os.environ["TF_INFERENCE_HISTORY_HOURS"] = "72"
os.environ["TF_USE_CALENDAR_FEATURES"] = "true"
os.environ["FORECAST_TRAINING_MODE"] = "maintenance"

module_path = ROOT / "services" / "forecast" / "main.py"
spec = importlib.util.spec_from_file_location("forecast_main_tf_service", module_path)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["forecast_main_tf_service"] = module
spec.loader.exec_module(module)

module.Base.metadata.create_all(bind=module.engine)

with module.SessionLocal() as db:
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    for i in range(900):
        ts = now - timedelta(minutes=900 - i)
        occ = int(round(42 + 11 * math.sin(i / 18) + 3 * math.cos(i / 57)))
        db.add(
            module.Count(
                ts=ts,
                zone_id="default-zone",
                occupancy=max(0, occ),
                utilization=max(0.0, occ / 100),
                source="tf-smoke",
                quality_score=0.95,
                quality_flags=["OK"],
            )
        )

    db.add(
        module.CalendarEvent(
            event_id="evt-smoke",
            zone_id="default-zone",
            title="Smoke Event",
            category="test",
            starts_at=now - timedelta(hours=2),
            ends_at=now + timedelta(hours=1),
            expected_impact=0.25,
            source="tf-smoke",
            metadata_json={"origin": "smoke"},
        )
    )
    db.commit()

with TestClient(module.app) as client:
    train = client.post(
        "/v1/train",
        json={"zone_id": "default-zone", "horizon": 12, "history_hours": 72, "full_retrain": False},
    )
    assert train.status_code == 200, train.text
    train_body = train.json()
    assert train_body["model_version"].startswith("tf-mlp-v2-")
    assert train_body["promotion_required"] is True
    module._set_model_promoted(train_body["paths"], promoted=True)

    status = client.get("/v1/model/status", params={"zone_id": "default-zone", "horizon": 12})
    assert status.status_code == 200, status.text
    status_body = status.json()
    assert status_body["exists"] is True

    forecast = client.get("/v1/forecast", params={"zone_id": "default-zone", "horizon": 12})
    assert forecast.status_code == 200, forecast.text
    body = forecast.json()
    assert body["model_version"].startswith("tf-mlp-v2-")
    assert len(body["points"]) == 12

print("forecast tf service smoke test passed")
