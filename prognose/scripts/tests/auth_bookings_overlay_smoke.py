#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
DB_FILE = ROOT / "auth_bookings_overlay_test.db"
if DB_FILE.exists():
    DB_FILE.unlink()

os.environ["DATABASE_URL"] = f"sqlite:///{DB_FILE}"
os.environ["DEFAULT_ZONE_ID"] = "default-zone"
os.environ["DEFAULT_ZONE_CAPACITY"] = "100"

api_gateway_dir = ROOT / "apps" / "api-gateway"
if str(api_gateway_dir) not in sys.path:
    sys.path.insert(0, str(api_gateway_dir))

module_path = api_gateway_dir / "main.py"
spec = importlib.util.spec_from_file_location("api_auth_overlay_main", module_path)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["api_auth_overlay_main"] = module
spec.loader.exec_module(module)


async def fake_fetch_forecast(*, zone_id: str, horizon: int) -> dict:
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    return {
        "zone_id": zone_id,
        "horizon": horizon,
        "generated_at": now.isoformat(),
        "summary": "Basisprognose ohne Buchungs-Overlay.",
        "model_version": "test-forecast-v1",
        "points": [
            {
                "timestamp": (now + timedelta(minutes=15)).isoformat(),
                "yhat": 10.0,
                "pi_low": 8.0,
                "pi_high": 12.0,
            },
            {
                "timestamp": (now + timedelta(minutes=45)).isoformat(),
                "yhat": 12.0,
                "pi_low": 9.0,
                "pi_high": 15.0,
            },
        ],
        "evidence": {
            "evidence_id": "ev-forecast-test",
            "generated_at": now.isoformat(),
            "time_window": {
                "from": (now - timedelta(minutes=60)).isoformat(),
                "to": now.isoformat(),
            },
            "sources": [{"type": "counts", "id": "forecast-window"}],
            "model": {"name": "forecast-test", "version": "v1"},
            "quality": {"score": 0.95, "flags": ["OK"]},
        },
        "lineage": {"product": "short_term"},
    }


async def fake_fetch_explanation(*, zone_id: str, horizon: int) -> dict:
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    return {
        "zone_id": zone_id,
        "horizon": horizon,
        "summary": "Die Prognose folgt vor allem dem aktuellen Trend.",
        "drivers": [
            {
                "name": "Trend",
                "impact": 0.8,
                "direction": "up",
                "description": "Die letzten Minuten zeigen steigende Belegung.",
            }
        ],
        "uncertainty": {"score": 0.2, "level": "low", "reason": "Ausreichend Verlauf vorhanden."},
        "evidence": {
            "evidence_id": "ev-explain-test",
            "generated_at": now.isoformat(),
            "time_window": {
                "from": (now - timedelta(minutes=60)).isoformat(),
                "to": now.isoformat(),
            },
            "sources": [{"type": "xai", "id": "explain-window"}],
            "model": {"name": "xai-test", "version": "v1"},
            "quality": {"score": 0.93, "flags": ["OK"]},
        },
    }


async def fake_fetch_recommendations(payload: dict) -> dict:
    return {
        "zone_id": payload["zone_id"],
        "horizon": payload["horizon"],
        "summary": "Empfehlungstest.",
        "actions": [],
        "gates": {"quality_ok": True, "uncertainty_ok": True, "notes": []},
        "evidence": {
            "evidence_id": "ev-reco-test",
            "generated_at": datetime.now(UTC).isoformat(),
            "time_window": {
                "from": datetime.now(UTC).isoformat(),
                "to": datetime.now(UTC).isoformat(),
            },
            "sources": [{"type": "recommendation", "id": "reco-test"}],
            "model": {"name": "reco-test", "version": "v1"},
            "quality": {"score": 0.9, "flags": ["OK"]},
        },
    }


module._fetch_forecast = fake_fetch_forecast
module._fetch_explanation = fake_fetch_explanation
module._fetch_recommendations = fake_fetch_recommendations

with TestClient(module.app) as client:
    me_initial = client.get("/api/v1/auth/me")
    assert me_initial.status_code == 200, me_initial.text
    assert me_initial.json()["authenticated"] is False

    invalid_role = client.post(
        "/api/v1/auth/register",
        json={"username": "BadRole", "password": "secret123", "role": "superuser"},
    )
    assert invalid_role.status_code == 422, invalid_role.text

    registered = client.post(
        "/api/v1/auth/register",
        json={"username": "Tester", "password": "secret123", "role": "admin"},
    )
    assert registered.status_code == 201, registered.text
    assert registered.json()["authenticated"] is True
    assert registered.json()["user"]["role"] == "admin"

    with module.SessionLocal() as db:
        created_user = (
            db.execute(
                module.select(module.User).where(module.User.username_normalized == "tester").limit(1)
            )
            .scalars()
            .first()
        )
        assert created_user is not None
        assert created_user.role == "admin"

    duplicate = client.post(
        "/api/v1/auth/register",
        json={"username": "tester", "password": "secret123", "role": "user"},
    )
    assert duplicate.status_code == 409, duplicate.text

    me_after_register = client.get("/api/v1/auth/me")
    assert me_after_register.status_code == 200, me_after_register.text
    assert me_after_register.json()["authenticated"] is True
    assert me_after_register.json()["user"]["role"] == "admin"

    invalid_booking = client.post(
        "/api/v1/bookings",
        json={
            "starts_at": (datetime.now(UTC) + timedelta(minutes=20)).isoformat(),
            "ends_at": (datetime.now(UTC) + timedelta(minutes=10)).isoformat(),
        },
    )
    assert invalid_booking.status_code == 400, invalid_booking.text

    created = client.post(
        "/api/v1/bookings",
        json={
            "starts_at": (datetime.now(UTC) + timedelta(minutes=10)).isoformat(),
            "ends_at": (datetime.now(UTC) + timedelta(minutes=50)).isoformat(),
        },
    )
    assert created.status_code == 201, created.text
    booking_id = created.json()["booking_id"]

    listed = client.get("/api/v1/bookings")
    assert listed.status_code == 200, listed.text
    assert len(listed.json()) == 1

    forecast = client.get("/api/v1/forecast/latest", params={"zone_id": "default-zone", "horizon": 60})
    assert forecast.status_code == 200, forecast.text
    forecast_payload = forecast.json()
    assert forecast_payload["points"][0]["yhat"] == 11.0, forecast_payload
    assert forecast_payload["points"][1]["pi_high"] == 16.0, forecast_payload

    explain = client.get("/api/v1/explain", params={"zone_id": "default-zone", "horizon": 60})
    assert explain.status_code == 200, explain.text
    explain_payload = explain.json()
    driver_names = [driver["name"] for driver in explain_payload["drivers"]]
    assert "Buchungen" in driver_names, explain_payload

    cancelled = client.delete(f"/api/v1/bookings/{booking_id}")
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["status"] == "cancelled"

    forecast_after_cancel = client.get("/api/v1/forecast/latest", params={"zone_id": "default-zone", "horizon": 60})
    assert forecast_after_cancel.status_code == 200, forecast_after_cancel.text
    assert forecast_after_cancel.json()["points"][0]["yhat"] == 10.0, forecast_after_cancel.text

    logged_out = client.post("/api/v1/auth/logout")
    assert logged_out.status_code == 200, logged_out.text
    assert logged_out.json()["authenticated"] is False

    me_after_logout = client.get("/api/v1/auth/me")
    assert me_after_logout.status_code == 200, me_after_logout.text
    assert me_after_logout.json()["authenticated"] is False

    registered_user = client.post(
        "/api/v1/auth/register",
        json={"username": "BasicUser", "password": "secret123", "role": "user"},
    )
    assert registered_user.status_code == 201, registered_user.text
    assert registered_user.json()["user"]["role"] == "user"

    with module.SessionLocal() as db:
        basic_user = (
            db.execute(
                module.select(module.User).where(module.User.username_normalized == "basicuser").limit(1)
            )
            .scalars()
            .first()
        )
        assert basic_user is not None
        assert basic_user.role == "user"

    logout_after_basic_user = client.post("/api/v1/auth/logout")
    assert logout_after_basic_user.status_code == 200, logout_after_basic_user.text
    assert logout_after_basic_user.json()["authenticated"] is False

    invalid_login = client.post(
        "/api/v1/auth/login",
        json={"username": "Tester", "password": "wrong"},
    )
    assert invalid_login.status_code == 401, invalid_login.text

    valid_login = client.post(
        "/api/v1/auth/login",
        json={"username": "tester", "password": "secret123"},
    )
    assert valid_login.status_code == 200, valid_login.text
    assert valid_login.json()["authenticated"] is True
    assert valid_login.json()["user"]["role"] == "admin"

print("auth/bookings/overlay smoke test passed")
