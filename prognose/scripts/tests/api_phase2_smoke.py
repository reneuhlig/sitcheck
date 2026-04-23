#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
DB_FILE = ROOT / "phase2_test.db"
if DB_FILE.exists():
    DB_FILE.unlink()

os.environ["DATABASE_URL"] = f"sqlite:///{DB_FILE}"
os.environ["DEFAULT_ZONE_ID"] = "default-zone"
os.environ["DEFAULT_ZONE_CAPACITY"] = "100"

module_path = ROOT / "apps" / "api-gateway" / "main.py"
spec = importlib.util.spec_from_file_location("api_main", module_path)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["api_main"] = module
spec.loader.exec_module(module)

with TestClient(module.app) as client:
    health = client.get("/health")
    assert health.status_code == 200, health.text

    zones = client.get("/api/v1/zones")
    assert zones.status_code == 200, zones.text
    payload = zones.json()
    assert payload and payload[0]["zone_id"] == "default-zone"

    now = datetime.now(UTC)
    ingest = {
        "points": [
            {
                "timestamp": now.isoformat(),
                "zone_id": "default-zone",
                "occupancy": 42,
                "source": "smoke-test",
                "quality_score": 0.95,
                "quality_flags": ["OK"],
                "evidence": {
                    "evidence_id": "ev-smoke",
                    "generated_at": now.isoformat(),
                    "time_window": {
                        "from": (now - timedelta(minutes=30)).isoformat(),
                        "to": now.isoformat(),
                    },
                    "sources": [{"type": "counts", "id": "smoke"}],
                    "model": {"name": "smoke", "version": "v1"},
                    "quality": {"score": 0.95, "flags": ["OK"]},
                },
            }
        ]
    }
    ingested = client.post("/api/v1/ingest/counts", json=ingest)
    assert ingested.status_code == 200, ingested.text
    assert ingested.json()["inserted"] == 1

    query_from = (now - timedelta(minutes=5)).isoformat()
    query_to = (now + timedelta(minutes=5)).isoformat()
    counts = client.get(
        "/api/v1/counts",
        params={"zone_id": "default-zone", "from": query_from, "to": query_to, "granularity": "raw"},
    )
    assert counts.status_code == 200, counts.text
    counts_body = counts.json()
    assert counts_body["points"], counts_body

    events = client.get(
        "/api/v1/calendar/events",
        params={"zone_id": "default-zone", "from": query_from, "to": (now + timedelta(days=2)).isoformat()},
    )
    assert events.status_code == 200, events.text
    assert isinstance(events.json(), list)

print("phase2 api smoke test passed")
