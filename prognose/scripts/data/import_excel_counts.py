#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill counts from Excel into sitcheck API")
    parser.add_argument("--file", default="/project_sitcheck/KI_Projekt_Daten_einJahr.xlsx")
    parser.add_argument("--sheet", default=0, help="Sheet name or index")
    parser.add_argument("--api-base-url", default=os.getenv("API_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--zone-id", default=os.getenv("DEFAULT_ZONE_ID", "default-zone"))
    parser.add_argument("--date-col", default="timestamp")
    parser.add_argument("--occupancy-col", default="occupancy")
    parser.add_argument("--utilization-col", default="utilization_pct")
    parser.add_argument("--capacity-col", default="capacity_effective")
    parser.add_argument("--source", default="excel-backfill")
    parser.add_argument("--quality-score", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--limit", type=int, default=0, help="Optional max rows after sorting (0=all)")
    parser.add_argument("--skip-reference-register", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=float, default=20.0)
    return parser.parse_args()


def _to_utc(ts: Any) -> datetime | None:
    try:
        parsed = pd.to_datetime(ts, errors="coerce", utc=True)
    except Exception:
        return None
    if parsed is None or pd.isna(parsed):
        return None
    return parsed.to_pydatetime().astimezone(UTC)


def _compute_utilization(row: pd.Series, util_col: str, capacity_col: str, occupancy: int) -> float:
    if util_col in row and pd.notna(row[util_col]):
        value = float(row[util_col])
        if value > 1.0:
            value /= 100.0
        return max(0.0, value)

    capacity = row.get(capacity_col, None)
    if capacity is not None and pd.notna(capacity) and float(capacity) > 0:
        return max(0.0, float(occupancy) / float(capacity))

    return 0.0


def _build_point(
    row: pd.Series,
    zone_id: str,
    date_col: str,
    occupancy_col: str,
    util_col: str,
    capacity_col: str,
    source: str,
    quality_score: float,
) -> dict[str, Any] | None:
    ts = _to_utc(row.get(date_col))
    if ts is None:
        return None

    occ_raw = row.get(occupancy_col)
    if occ_raw is None or pd.isna(occ_raw):
        return None

    occupancy = max(0, int(round(float(occ_raw))))
    utilization = _compute_utilization(row, util_col, capacity_col, occupancy)

    evidence = {
        "evidence_id": f"excel-{uuid.uuid4()}",
        "generated_at": datetime.now(UTC).isoformat(),
        "time_window": {
            "from": (ts - timedelta(minutes=1)).isoformat(),
            "to": ts.isoformat(),
        },
        "sources": [
            {
                "type": "excel",
                "id": source,
                "note": "historical backfill",
            }
        ],
        "model": {
            "name": "excel_backfill",
            "version": "v1",
            "backend": "batch",
        },
        "quality": {
            "score": max(0.0, min(1.0, quality_score)),
            "flags": ["HISTORICAL_BACKFILL"],
        },
    }

    return {
        "timestamp": ts.isoformat(),
        "zone_id": zone_id,
        "occupancy": occupancy,
        "utilization": utilization,
        "source": source,
        "quality_score": max(0.0, min(1.0, quality_score)),
        "quality_flags": ["HISTORICAL_BACKFILL"],
        "evidence": evidence,
    }


def _chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = _parse_args()

    file_path = Path(args.file).expanduser().resolve()
    if not file_path.exists():
        print(f"[error] Excel file not found: {file_path}", file=sys.stderr)
        return 2
    file_checksum = _sha256_file(file_path)

    try:
        frame = pd.read_excel(file_path, sheet_name=args.sheet, engine="openpyxl")
    except ImportError:
        print("[error] Missing dependency 'openpyxl'. Install it first (e.g. pip install openpyxl).", file=sys.stderr)
        return 2

    required = [args.date_col, args.occupancy_col]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        print(f"[error] Missing columns: {missing}. Available: {list(frame.columns)}", file=sys.stderr)
        return 2

    frame = frame.sort_values(args.date_col)
    if args.limit > 0:
        frame = frame.tail(args.limit)

    points: list[dict[str, Any]] = []
    skipped = 0
    for _, row in frame.iterrows():
        point = _build_point(
            row=row,
            zone_id=args.zone_id,
            date_col=args.date_col,
            occupancy_col=args.occupancy_col,
            util_col=args.utilization_col,
            capacity_col=args.capacity_col,
            source=args.source,
            quality_score=args.quality_score,
        )
        if point is None:
            skipped += 1
            continue
        points.append(point)

    if not points:
        print("[error] No valid points after parsing.", file=sys.stderr)
        return 2

    print(
        f"[info] prepared={len(points)} skipped={skipped} range={points[0]['timestamp']} .. {points[-1]['timestamp']}"
    )

    source_metadata = {
        "checksum": file_checksum,
        "uri_or_path": str(file_path),
        "label": file_path.name,
        "sheet": args.sheet,
        "time_from": points[0]["timestamp"],
        "time_to": points[-1]["timestamp"],
        "row_count": len(points),
    }

    if args.dry_run:
        print("[info] dry-run enabled, no API requests sent.")
        return 0

    if not args.skip_reference_register:
        register_payload = {
            "zone_id": args.zone_id,
            "source_type": file_path.suffix.lstrip(".").lower() or "file",
            "label": file_path.name,
            "uri_or_path": str(file_path),
            "checksum": file_checksum,
            "imported_at": datetime.now(UTC).isoformat(),
            "time_from": points[0]["timestamp"],
            "time_to": points[-1]["timestamp"],
            "row_count": len(points),
            "ingest_job_id": f"excel-import-{uuid.uuid4().hex[:8]}",
            "metadata": {
                "sheet": args.sheet,
                "source": args.source,
                "date_col": args.date_col,
                "occupancy_col": args.occupancy_col,
                "utilization_col": args.utilization_col,
                "capacity_col": args.capacity_col,
            },
        }
        try:
            ref_response = requests.post(
                f"{args.api_base_url.rstrip('/')}/api/v1/references/training-data/register",
                json=register_payload,
                timeout=args.timeout,
            )
            ref_response.raise_for_status()
            ref_payload = ref_response.json() if ref_response.content else {}
            reference = ref_payload.get("reference", {}) if isinstance(ref_payload, dict) else {}
            source_metadata["reference_id"] = reference.get("reference_id")
            print(f"[info] registered training-data reference for {file_path.name}")
        except requests.RequestException as exc:
            print(f"[warn] reference registration failed: {exc}", file=sys.stderr)

    for point in points:
        try:
            source = point["evidence"]["sources"][0]
            source["uri"] = str(file_path)
            source["metadata"] = dict(source_metadata)
        except Exception:
            continue

    endpoint = f"{args.api_base_url.rstrip('/')}/api/v1/ingest/counts"
    batches = _chunked(points, max(1, args.batch_size))
    sent = 0

    for idx, chunk in enumerate(batches, start=1):
        payload = {"points": chunk}
        try:
            response = requests.post(endpoint, json=payload, timeout=args.timeout)
            response.raise_for_status()
            body = response.json()
            inserted = int(body.get("inserted", len(chunk)))
            sent += inserted
            print(f"[info] batch {idx}/{len(batches)} inserted={inserted}")
        except requests.RequestException as exc:
            print(f"[error] batch {idx} failed: {exc}", file=sys.stderr)
            return 1

    print(f"[done] inserted_total={sent} points_total={len(points)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
