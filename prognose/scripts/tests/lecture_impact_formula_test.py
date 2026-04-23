#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path


def _load_lecture_ingest_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "services" / "lecture-ingest" / "main.py"
    spec = importlib.util.spec_from_file_location("sitcheck_lecture_ingest_main", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _find_row(rows, stamp: datetime):
    for row in rows:
        if row["ts"] == stamp:
            return row
    raise AssertionError(f"timestamp not found in aggregate rows: {stamp.isoformat()}")


def _assert_close(actual: float, expected: float, eps: float = 1e-6):
    if abs(actual - expected) > eps:
        raise AssertionError(f"value mismatch: actual={actual} expected={expected}")


def main() -> int:
    module = _load_lecture_ingest_module()

    module.LECTURE_EFFECT_ENABLED = True
    module.LECTURE_AVG_ATTENDANCE = 20.0
    module.LECTURE_HEAVY_BIB_PERSONS = 4.0
    module.LECTURE_HEAVY_WINDOW_MINUTES = 60
    module.LECTURE_HEAVY_PHRASES = [
        "logik und algebra",
        "operations research",
        "grundlagen und logik",
        "maschinelles lernen",
        "machine learning",
        "deep learning",
        "digitale signalverarbeitung",
    ]
    module.LECTURE_HEAVY_KEYWORDS = [
        "mathe",
        "mathematik",
        "statistik",
        "algorithm",
        "theorie",
        "theoret",
        "physik",
        "analysis",
        "lineare algebra",
        "regelungstechnik",
    ]
    module.LECTURE_ONSITE_TYPES = {"PRESENCE"}

    assert module._is_heavy_module({"name": "Ingenieur-Mathematik I"})
    assert module._is_heavy_module({"name": "Algorithmen und Datenstrukturen"})
    assert module._is_heavy_module({"name": "Operations Research"})
    assert module._is_heavy_module({"name": "Logik und Algebra"})
    assert module._is_heavy_module({"name": "Regelungstechnik 1"})
    assert module._is_heavy_module({"name": "Maschinelles Lernen"})
    assert module._is_heavy_module({"name": "Advanced Machine Learning"})
    assert module._is_heavy_module({"name": "Neuronale Netze und Deep Learning"})
    assert not module._is_heavy_module({"name": "Englisch Grundlagen"})
    assert not module._is_heavy_module({"name": "Funktionen & Systeme der Logistik: Handelslogik / eCommerce"})

    raw_events = [
        {
            "start": datetime(2026, 1, 8, 10, 0, tzinfo=UTC),
            "end": datetime(2026, 1, 8, 11, 0, tzinfo=UTC),
            "course": "TINF24A",
            "name": "Ingenieur-Mathematik I",
            "type": "PRESENCE",
        },
        {
            "start": datetime(2026, 1, 8, 10, 0, tzinfo=UTC),
            "end": datetime(2026, 1, 8, 11, 0, tzinfo=UTC),
            "course": "TINF24A",
            "name": "Software Engineering",
            "type": "PRESENCE",
        },
        {
            "start": datetime(2026, 1, 8, 10, 0, tzinfo=UTC),
            "end": datetime(2026, 1, 8, 11, 0, tzinfo=UTC),
            "course": "TINF24A",
            "name": "Online Q&A",
            "type": "ONLINE",
        },
    ]

    onsite_events = module._filter_onsite_events(raw_events)
    if len(onsite_events) != 2:
        raise AssertionError(f"expected 2 onsite events, got {len(onsite_events)}")

    rows = module._aggregate_activity(
        events=onsite_events,
        source="unit-test",
        quality_score=1.0,
        quality_flags=["TEST"],
    )
    if not rows:
        raise AssertionError("aggregate produced no rows")

    row_1030 = _find_row(rows, datetime(2026, 1, 8, 10, 30, tzinfo=UTC))
    meta_1030 = row_1030["metadata"]
    _assert_close(meta_1030["lecture_pull_regular"], 40.0)
    _assert_close(meta_1030["heavy_bib_bonus"], 4.0)
    _assert_close(meta_1030["lecture_net_pull"], 36.0)
    if meta_1030["heavy_active_lectures"] != 1:
        raise AssertionError("heavy_active_lectures at 10:30 should be 1")
    if meta_1030["heavy_ended_last_60m"] != 0:
        raise AssertionError("heavy_ended_last_60m at 10:30 should be 0")

    row_1130 = _find_row(rows, datetime(2026, 1, 8, 11, 30, tzinfo=UTC))
    meta_1130 = row_1130["metadata"]
    _assert_close(meta_1130["lecture_pull_regular"], 0.0)
    _assert_close(meta_1130["heavy_bib_bonus"], 4.0)
    _assert_close(meta_1130["lecture_net_pull"], -4.0)
    if meta_1130["heavy_active_lectures"] != 0:
        raise AssertionError("heavy_active_lectures at 11:30 should be 0")
    if meta_1130["heavy_ended_last_60m"] != 1:
        raise AssertionError("heavy_ended_last_60m at 11:30 should be 1")

    row_1200 = _find_row(rows, datetime(2026, 1, 8, 12, 0, tzinfo=UTC))
    meta_1200 = row_1200["metadata"]
    if meta_1200["heavy_ended_last_60m"] != 0:
        raise AssertionError("heavy_ended_last_60m at 12:00 should be 0")
    _assert_close(meta_1200["lecture_net_pull"], 0.0)

    print("lecture impact formula tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
