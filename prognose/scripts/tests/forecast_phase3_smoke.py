#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import math
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
DB_FILE = ROOT / "phase3_forecast_test.db"
if DB_FILE.exists():
    DB_FILE.unlink()

os.environ["DATABASE_URL"] = f"sqlite:///{DB_FILE}"
os.environ["FORECAST_MODEL_BACKEND"] = "baseline"

module_path = ROOT / "services" / "forecast" / "main.py"
spec = importlib.util.spec_from_file_location("forecast_main", module_path)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["forecast_main"] = module
spec.loader.exec_module(module)

module.Base.metadata.create_all(bind=module.engine)

with module.SessionLocal() as db:
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    for i in range(180):
        ts = now - timedelta(minutes=180 - i)
        occ = int(round(45 + 10 * math.sin(i / 10)))
        db.add(
            module.Count(
                ts=ts,
                zone_id="default-zone",
                occupancy=max(0, occ),
                utilization=max(0.0, occ / 100),
                source="smoke",
                quality_score=0.95,
            )
        )
    db.commit()

with TestClient(module.app) as client:
    response = client.get("/v1/forecast", params={"zone_id": "default-zone", "horizon": 60})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["zone_id"] == "default-zone"
    assert body["horizon"] == 60
    assert body["model_version"]
    assert len(body["points"]) == 60
    assert {"yhat", "pi_low", "pi_high"}.issubset(body["points"][0].keys())
    evidence = body.get("evidence", {})
    sources = evidence.get("sources", [])
    assert any(str(src.get("id", "")).startswith("window:") for src in sources), "missing counts window context source"
    quality = evidence.get("quality", {})
    assert "flags" in quality and isinstance(quality["flags"], list), "missing quality flags"

print("phase3 forecast smoke test passed")
