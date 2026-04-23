#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import math
import os
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
DB_FILE = ROOT / "phase5_chain_test.db"
if DB_FILE.exists():
    DB_FILE.unlink()

os.environ["DATABASE_URL"] = f"sqlite:///{DB_FILE}"
os.environ["DEFAULT_ZONE_ID"] = "default-zone"
os.environ["DEFAULT_ZONE_CAPACITY"] = "100"
os.environ["FORECAST_SERVICE_URL"] = "http://127.0.0.1:65530"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


api = load_module("api_main_phase5", ROOT / "apps" / "api-gateway" / "main.py")
forecast = load_module("forecast_main_phase5", ROOT / "services" / "forecast" / "main.py")
xai = load_module("xai_main_phase5", ROOT / "services" / "xai" / "main.py")
rec = load_module("rec_main_phase5", ROOT / "services" / "recommendations" / "main.py")

api.Base.metadata.create_all(bind=api.engine)
forecast.Base.metadata.create_all(bind=forecast.engine)
xai.Base.metadata.create_all(bind=xai.engine)

now = datetime.now(UTC).replace(second=0, microsecond=0)

with TestClient(api.app) as api_client:
    for i in range(180):
        ts = now - timedelta(minutes=180 - i)
        occ = int(round(45 + 10 * math.sin(i / 9)))
        payload = {
            "points": [
                {
                    "timestamp": ts.isoformat(),
                    "zone_id": "default-zone",
                    "occupancy": max(0, occ),
                    "source": "phase5-chain",
                    "quality_score": 0.92,
                    "quality_flags": ["OK"],
                    "evidence": {
                        "evidence_id": f"ev-{i}",
                        "generated_at": ts.isoformat(),
                        "time_window": {
                            "from": (ts - timedelta(minutes=30)).isoformat(),
                            "to": ts.isoformat(),
                        },
                        "sources": [{"type": "counts", "id": f"pt-{i}"}],
                        "model": {"name": "chain", "version": "v1"},
                        "quality": {"score": 0.92, "flags": ["OK"]},
                    },
                }
            ]
        }
        r = api_client.post("/api/v1/ingest/counts", json=payload)
        assert r.status_code == 200, r.text

with xai.SessionLocal() as db:
    db.add(
        xai.CalendarEvent(
            event_id=f"evt-{uuid.uuid4()}",
            zone_id="default-zone",
            title="Peak Event",
            category="campus-flow",
            starts_at=now + timedelta(minutes=10),
            ends_at=now + timedelta(minutes=80),
            expected_impact=0.25,
            source="mock",
        )
    )
    db.commit()

with TestClient(forecast.app) as forecast_client, TestClient(xai.app) as xai_client, TestClient(rec.app) as rec_client:
    forecast_resp = forecast_client.get("/v1/forecast", params={"zone_id": "default-zone", "horizon": 60})
    assert forecast_resp.status_code == 200, forecast_resp.text

    xai_resp = xai_client.get("/v1/explain", params={"zone_id": "default-zone", "horizon": 60})
    assert xai_resp.status_code == 200, xai_resp.text

    rec_resp = rec_client.post(
        "/v1/recommendations",
        json={
            "zone_id": "default-zone",
            "horizon": 60,
            "capacity": 100,
            "forecast": forecast_resp.json(),
            "explanation": xai_resp.json(),
        },
    )
    assert rec_resp.status_code == 200, rec_resp.text
    assert rec_resp.json()["gates"]["quality_ok"] is True

print("phase5 e2e chain smoke test passed")
