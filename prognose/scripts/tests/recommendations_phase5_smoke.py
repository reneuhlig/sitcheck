#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
module_path = ROOT / "services" / "recommendations" / "main.py"
spec = importlib.util.spec_from_file_location("rec_main", module_path)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["rec_main"] = module
spec.loader.exec_module(module)

forecast = {
    "points": [{"timestamp": "2026-02-18T19:01:00Z", "yhat": 92.0, "pi_low": 85.0, "pi_high": 100.0}]
}
explanation = {
    "uncertainty": {"score": 0.3, "level": "low", "reason": "stable"},
    "evidence": {"quality": {"score": 0.9, "flags": ["OK"]}},
}

with TestClient(module.app) as client:
    rec_response = client.post(
        "/v1/recommendations",
        json={
            "zone_id": "default-zone",
            "horizon": 60,
            "capacity": 100,
            "forecast": forecast,
            "explanation": explanation,
        },
    )
    assert rec_response.status_code == 200, rec_response.text
    rec_body = rec_response.json()
    assert rec_body["actions"], rec_body
    assert rec_body["gates"]["quality_ok"] is True

    sim_response = client.post(
        "/v1/scenarios/simulate",
        json={
            "zone_id": "default-zone",
            "horizon": 60,
            "persist": False,
            "changes": {"open_room": True, "push_time_minutes": 10},
            "capacity": 100,
            "forecast": forecast,
            "explanation": explanation,
        },
    )
    assert sim_response.status_code == 200, sim_response.text
    sim_body = sim_response.json()
    assert sim_body["counterfactual"]["peak_occupancy"] < sim_body["baseline"]["peak_occupancy"]

print("phase5 recommendations smoke test passed")
