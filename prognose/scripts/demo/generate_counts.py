#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import random
import time
from datetime import UTC, datetime, timedelta

import requests


def build_payload(zone_id: str, occupancy: int, quality: float) -> dict:
    now = datetime.now(UTC)
    evidence = {
        "evidence_id": f"demo-{int(now.timestamp())}",
        "generated_at": now.isoformat(),
        "time_window": {
            "from": (now - timedelta(minutes=30)).isoformat(),
            "to": now.isoformat(),
        },
        "sources": [{"type": "counts", "id": "demo-generator", "note": "synthetic"}],
        "model": {"name": "demo-generator", "version": "v1"},
        "quality": {"score": quality, "flags": ["SYNTHETIC"]},
    }
    return {
        "points": [
            {
                "timestamp": now.isoformat(),
                "zone_id": zone_id,
                "occupancy": occupancy,
                "source": "demo-generator",
                "quality_score": quality,
                "quality_flags": ["SYNTHETIC"],
                "evidence": evidence,
            }
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic counts for sitcheck API")
    parser.add_argument("--api-base-url", default=os.getenv("API_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--zone-id", default=os.getenv("ZONE_ID", "default-zone"))
    parser.add_argument("--iterations", type=int, default=int(os.getenv("DEMO_ITERATIONS", "120")))
    parser.add_argument("--interval", type=float, default=float(os.getenv("DEMO_INTERVAL_SECONDS", "1.0")))
    parser.add_argument("--base", type=int, default=35)
    parser.add_argument("--amplitude", type=int, default=12)
    args = parser.parse_args()

    endpoint = f"{args.api_base_url.rstrip('/')}/api/v1/ingest/counts"
    print(f"[demo] target={endpoint} zone={args.zone_id} iterations={args.iterations}")

    for i in range(args.iterations):
        wave = args.base + args.amplitude * math.sin(i / 8)
        noise = random.randint(-3, 3)
        occupancy = max(0, int(round(wave + noise)))
        quality = min(1.0, max(0.5, 0.92 + random.uniform(-0.05, 0.05)))

        payload = build_payload(args.zone_id, occupancy, quality)
        try:
            response = requests.post(endpoint, json=payload, timeout=10)
            response.raise_for_status()
            print(f"[demo] {i+1:04d}/{args.iterations} occupancy={occupancy} status={response.status_code}")
        except requests.RequestException as exc:
            print(f"[demo] request failed on step {i+1}: {exc}")

        if i < args.iterations - 1:
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
