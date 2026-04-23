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
DB_FILE = ROOT / "phase4_xai_test.db"
if DB_FILE.exists():
    DB_FILE.unlink()

os.environ["DATABASE_URL"] = f"sqlite:///{DB_FILE}"
os.environ["FORECAST_SERVICE_URL"] = "http://127.0.0.1:65530"  # force fallback path

module_path = ROOT / "services" / "xai" / "main.py"
spec = importlib.util.spec_from_file_location("xai_main", module_path)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["xai_main"] = module
spec.loader.exec_module(module)

module.Base.metadata.create_all(bind=module.engine)

with module.SessionLocal() as db:
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    for i in range(180):
        ts = now - timedelta(minutes=180 - i)
        occ = int(round(50 + 8 * math.sin(i / 8)))
        db.add(
            module.Count(
                ts=ts,
                zone_id="default-zone",
                occupancy=max(0, occ),
                utilization=max(0.0, occ / 100),
                source="smoke",
                quality_score=0.9,
            )
        )

    db.add(
        module.CalendarEvent(
            event_id=f"evt-{uuid.uuid4()}",
            zone_id="default-zone",
            title="Test Event",
            category="campus-flow",
            starts_at=now + timedelta(minutes=10),
            ends_at=now + timedelta(minutes=80),
            expected_impact=0.3,
            source="mock",
        )
    )
    db.commit()

with TestClient(module.app) as client:
    response = client.get("/v1/explain", params={"zone_id": "default-zone", "horizon": 60})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["zone_id"] == "default-zone"
    assert body["drivers"]
    assert body["summary"]
    assert "evidence" in body and body["evidence"]["sources"]
    assert "uncertainty" in body and "score" in body["uncertainty"]

print("phase4 xai smoke test passed")
