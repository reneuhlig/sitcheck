#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from jsonschema import Draft202012Validator, RefResolver


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "packages" / "shared" / "schemas"


def load_schemas() -> dict[str, dict[str, Any]]:
    store: dict[str, dict[str, Any]] = {}
    for path in SCHEMA_DIR.glob("*.schema.json"):
        schema = json.loads(path.read_text())
        store[schema["$id"]] = schema
    return store


def validate_with_schema(schema_id: str, payload: Any, store: dict[str, dict[str, Any]]) -> None:
    schema = store[schema_id]
    resolver = RefResolver.from_schema(schema, store=store)
    validator = Draft202012Validator(schema, resolver=resolver)
    validator.validate(payload)


def ingest_seed(base_url: str, zone_id: str, points: int = 40) -> int:
    now = datetime.now(UTC)
    inserted = 0
    for i in range(points):
        ts = (now - timedelta(minutes=points - i)).isoformat()
        occ = max(0, int(round(35 + 10 * math.sin(i / 6))))
        payload = {
            "points": [
                {
                    "timestamp": ts,
                    "zone_id": zone_id,
                    "occupancy": occ,
                    "source": "api-schema-smoke",
                    "quality_score": 0.91,
                    "quality_flags": ["OK"],
                    "evidence": {
                        "evidence_id": f"api-schema-{i}",
                        "generated_at": ts,
                        "time_window": {
                            "from": (now - timedelta(minutes=60)).isoformat(),
                            "to": ts,
                        },
                        "sources": [{"type": "counts", "id": f"api-schema-{i}"}],
                        "model": {"name": "seed", "version": "v1"},
                        "quality": {"score": 0.91, "flags": ["OK"]},
                    },
                }
            ]
        }
        res = requests.post(f"{base_url}/api/v1/ingest/counts", json=payload, timeout=15)
        res.raise_for_status()
        inserted += int(res.json().get("inserted", 0))
    return inserted


def create_snapshot(base_url: str, zone_id: str, horizon: int, internal_token: str) -> dict[str, Any]:
    if not internal_token:
        raise RuntimeError("INTERNAL_API_TOKEN must be set for snapshot smoke checks")
    res = requests.post(
        f"{base_url}/api/v1/internal/forecast/snapshot",
        json={"zone_id": zone_id, "horizon": horizon},
        headers={"X-Internal-Token": internal_token},
        timeout=30,
    )
    res.raise_for_status()
    return res.json()


def main() -> int:
    parser = argparse.ArgumentParser(description="Sitcheck API schema smoke test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--zone-id", default="default-zone")
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--internal-token", default=os.getenv("INTERNAL_API_TOKEN", ""))
    args = parser.parse_args()

    store = load_schemas()

    health = requests.get(f"{args.base_url}/health", timeout=15)
    health.raise_for_status()

    zones_res = requests.get(f"{args.base_url}/api/v1/zones", timeout=15)
    zones_res.raise_for_status()
    zones = zones_res.json()
    if not isinstance(zones, list) or not zones:
        raise RuntimeError("/api/v1/zones returned empty or invalid response")

    for zone in zones:
        validate_with_schema("https://sitcheck.dev/schemas/zone.schema.json", zone, store)

    if not any(z.get("zone_id") == args.zone_id for z in zones):
        raise RuntimeError(f"zone_id={args.zone_id} not found in /zones")

    inserted = ingest_seed(args.base_url, args.zone_id, points=40)

    now = datetime.now(UTC)
    from_ts = (now - timedelta(hours=3)).isoformat()
    to_ts = (now + timedelta(minutes=5)).isoformat()

    counts_res = requests.get(
        f"{args.base_url}/api/v1/counts",
        params={"zone_id": args.zone_id, "from": from_ts, "to": to_ts, "granularity": "1m"},
        timeout=15,
    )
    counts_res.raise_for_status()
    counts_payload = counts_res.json()

    points = counts_payload.get("points", [])
    if not points:
        raise RuntimeError("/api/v1/counts returned no points after seeding")

    for point in points:
        validate_with_schema("https://sitcheck.dev/schemas/count-point.schema.json", point, store)

    lecture_res = requests.get(
        f"{args.base_url}/api/v1/lectures/activity",
        params={"zone_id": args.zone_id, "from": from_ts, "to": to_ts, "granularity": "1m"},
        timeout=15,
    )
    lecture_res.raise_for_status()
    lecture_payload = lecture_res.json()
    validate_with_schema("https://sitcheck.dev/schemas/lecture-activity-response.schema.json", lecture_payload, store)

    forecast_res = requests.get(
        f"{args.base_url}/api/v1/forecast",
        params={"zone_id": args.zone_id, "horizon": args.horizon},
        timeout=20,
    )
    forecast_res.raise_for_status()
    forecast_payload = forecast_res.json()
    validate_with_schema("https://sitcheck.dev/schemas/forecast-response.schema.json", forecast_payload, store)

    snapshot_payload = create_snapshot(
        base_url=args.base_url,
        zone_id=args.zone_id,
        horizon=args.horizon,
        internal_token=args.internal_token or "",
    )
    validate_with_schema("https://sitcheck.dev/schemas/forecast-latest-response.schema.json", snapshot_payload, store)

    latest_res = requests.get(
        f"{args.base_url}/api/v1/forecast/latest",
        params={"zone_id": args.zone_id, "horizon": args.horizon},
        timeout=20,
    )
    latest_res.raise_for_status()
    latest_payload = latest_res.json()
    validate_with_schema("https://sitcheck.dev/schemas/forecast-latest-response.schema.json", latest_payload, store)

    weekly_latest_res = requests.get(
        f"{args.base_url}/api/v1/forecast/weekly/latest",
        params={"zone_id": args.zone_id, "days": 7, "slot_minutes": 60},
        timeout=25,
    )
    weekly_latest_res.raise_for_status()
    weekly_latest_payload = weekly_latest_res.json()
    validate_with_schema("https://sitcheck.dev/schemas/weekly-forecast-response.schema.json", weekly_latest_payload, store)

    command_center_res = requests.get(
        f"{args.base_url}/api/v1/dashboard/command-center",
        params={
            "zone_id": args.zone_id,
            "horizon": args.horizon,
            "history_minutes": 180,
            "stale_seconds": 900,
            "long_term_days": 14,
        },
        timeout=30,
    )
    command_center_res.raise_for_status()
    command_center_payload = command_center_res.json()
    validate_with_schema("https://sitcheck.dev/schemas/dashboard-command-center.schema.json", command_center_payload, store)

    weekly_explain_res = requests.get(
        f"{args.base_url}/api/v1/explain/weekly",
        params={"zone_id": args.zone_id, "days": 7, "slot_minutes": 60},
        timeout=30,
    )
    weekly_explain_res.raise_for_status()
    weekly_explain_payload = weekly_explain_res.json()
    validate_with_schema("https://sitcheck.dev/schemas/weekly-explain-response.schema.json", weekly_explain_payload, store)

    lineage_res = requests.get(
        f"{args.base_url}/api/v1/models/lineage/latest",
        params={"zone_id": args.zone_id, "product": "short_term", "horizon": args.horizon},
        timeout=20,
    )
    if lineage_res.status_code == 200:
        validate_with_schema("https://sitcheck.dev/schemas/model-lineage.schema.json", lineage_res.json(), store)

    history_res = requests.get(
        f"{args.base_url}/api/v1/forecast/history",
        params={
            "zone_id": args.zone_id,
            "horizon": args.horizon,
            "from": (now - timedelta(hours=6)).isoformat(),
            "to": now.isoformat(),
            "limit": 20,
        },
        timeout=20,
    )
    history_res.raise_for_status()
    history_payload = history_res.json()
    history_items = history_payload.get("items", [])
    if not isinstance(history_items, list):
        raise RuntimeError("/api/v1/forecast/history returned invalid items payload")
    for item in history_items:
        validate_with_schema("https://sitcheck.dev/schemas/forecast-latest-response.schema.json", item, store)

    weekly_history_res = requests.get(
        f"{args.base_url}/api/v1/forecast/weekly/history",
        params={
            "zone_id": args.zone_id,
            "days": 7,
            "slot_minutes": 60,
            "from": (now - timedelta(days=7)).isoformat(),
            "to": now.isoformat(),
            "limit": 10,
        },
        timeout=20,
    )
    weekly_history_res.raise_for_status()
    weekly_history_payload = weekly_history_res.json()
    weekly_history_items = weekly_history_payload.get("items", [])
    if not isinstance(weekly_history_items, list):
        raise RuntimeError("/api/v1/forecast/weekly/history returned invalid items payload")
    for item in weekly_history_items:
        validate_with_schema("https://sitcheck.dev/schemas/weekly-forecast-response.schema.json", item, store)

    explain_res = requests.get(
        f"{args.base_url}/api/v1/explain",
        params={"zone_id": args.zone_id, "horizon": args.horizon},
        timeout=20,
    )
    explain_res.raise_for_status()
    explain_payload = explain_res.json()
    validate_with_schema("https://sitcheck.dev/schemas/explanation.schema.json", explain_payload, store)

    explain_context_res = requests.get(
        f"{args.base_url}/api/v1/explain/context",
        params={
            "zone_id": args.zone_id,
            "horizon": args.horizon,
            "audience": "ops",
            "language": "de",
            "query": "api-smoke",
        },
        timeout=30,
    )
    explain_context_res.raise_for_status()
    explain_context_payload = explain_context_res.json()
    context = explain_context_payload.get("context", {})
    validate_with_schema("https://sitcheck.dev/schemas/llm-explainability-context-v2.schema.json", context, store)

    explain_narrative_res = requests.post(
        f"{args.base_url}/api/v1/explain/narrative",
        json={
            "zone_id": args.zone_id,
            "horizon": args.horizon,
            "audience": "ops",
            "query": "api-smoke",
            "language": "de",
            "response_mode": "free",
        },
        timeout=45,
    )
    explain_narrative_res.raise_for_status()
    explain_narrative_payload = explain_narrative_res.json()
    validate_with_schema(
        "https://sitcheck.dev/schemas/llm-explanation-response.schema.json",
        explain_narrative_payload.get("response", {}),
        store,
    )

    rec_res = requests.get(
        f"{args.base_url}/api/v1/recommendations",
        params={"zone_id": args.zone_id, "horizon": args.horizon},
        timeout=20,
    )
    rec_res.raise_for_status()
    rec_payload = rec_res.json()
    validate_with_schema("https://sitcheck.dev/schemas/recommendation.schema.json", rec_payload, store)

    scenario_in = {
        "zone_id": args.zone_id,
        "horizon": args.horizon,
        "persist": False,
        "changes": {"open_room": True, "push_time_minutes": 5},
    }

    scenario_res = requests.post(
        f"{args.base_url}/api/v1/scenarios/simulate",
        json=scenario_in,
        timeout=20,
    )
    scenario_res.raise_for_status()
    scenario_payload = scenario_res.json()
    validate_with_schema("https://sitcheck.dev/schemas/scenario-result.schema.json", scenario_payload, store)

    cal_res = requests.get(
        f"{args.base_url}/api/v1/calendar/events",
        params={"zone_id": args.zone_id, "from": from_ts, "to": to_ts},
        timeout=15,
    )
    cal_res.raise_for_status()
    cal_payload = cal_res.json()
    if not isinstance(cal_payload, list):
        raise RuntimeError("/api/v1/calendar/events did not return list")
    for event in cal_payload:
        validate_with_schema("https://sitcheck.dev/schemas/calendar-event.schema.json", event, store)

    print("api_schema_smoke passed")
    print(
        json.dumps(
            {
                "seed_inserted": inserted,
                "counts_points": len(points),
                "lecture_points": len(lecture_payload.get("points", [])),
                "forecast_points": len(forecast_payload.get("points", [])),
                "latest_source": latest_payload.get("source"),
                "command_center_alerts": len(command_center_payload.get("alerts", [])),
                "forecast_history_items": len(history_items),
                "drivers": len(explain_payload.get("drivers", [])),
                "narrative_mode": explain_narrative_payload.get("mode"),
                "actions": len(rec_payload.get("actions", [])),
                "calendar_events": len(cal_payload),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
