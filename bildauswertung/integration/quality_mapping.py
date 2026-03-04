from __future__ import annotations

from typing import Any, Iterable, List, Tuple


def map_tracking_quality(
    *,
    track_ok: bool,
    track_error: str,
    tracks: Iterable[dict[str, Any]] | None,
    run_tracking_now: bool,
) -> Tuple[float, List[str]]:
    """Map tracking runtime signals to prognose-compatible quality values."""
    flags: List[str] = []
    score = 1.0
    track_items = list(tracks or [])

    if not track_ok:
        flags.append("TRACK_ERROR")
        score = min(score, 0.20)
    if track_error:
        flags.append("TRACK_ERROR_DETAIL")
        score = min(score, 0.35)

    confidences: List[float] = []
    stale_count = 0
    for item in track_items:
        try:
            conf = float(item.get("confidence", 0.0))
            if conf >= 0:
                confidences.append(conf)
        except (TypeError, ValueError):
            continue
        if bool(item.get("is_stale", False)):
            stale_count += 1

    if not track_items:
        flags.append("NO_TRACKS")
        score = min(score, 0.85)

    if confidences:
        avg_conf = sum(confidences) / max(1, len(confidences))
        if avg_conf < 0.40:
            flags.append("LOW_TRACK_CONF")
            score = min(score, max(0.30, avg_conf))
        elif avg_conf < 0.60:
            flags.append("MID_TRACK_CONF")
            score = min(score, max(0.50, avg_conf))
        else:
            score = min(score, min(1.0, avg_conf))
    elif track_items:
        flags.append("NO_TRACK_CONF")
        score = min(score, 0.75)

    if track_items:
        stale_ratio = stale_count / float(len(track_items))
        if stale_ratio >= 0.80:
            flags.append("TRACK_STALE_HIGH")
            score = min(score, 0.55)
        elif stale_ratio >= 0.50:
            flags.append("TRACK_STALE_MEDIUM")
            score = min(score, 0.70)

    if not run_tracking_now:
        flags.append("TRACK_REUSE")
        score = min(score, 0.90)

    if not flags:
        flags = ["OK"]
    else:
        flags = _dedupe_preserve_order(flags)
        if "TRACK_ERROR" not in flags and score >= 0.95:
            flags = ["OK"]

    return round(max(0.0, min(1.0, score)), 4), flags


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for value in values:
        key = str(value).upper()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped

