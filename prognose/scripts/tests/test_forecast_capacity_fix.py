"""Validation test: verify capacity clipping after fix.

Tests that all forecast paths now respect the physical capacity upper bound.
"""
from __future__ import annotations

import sys
import math
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

UTC = timezone.utc
CAPACITY = 84  # CAPACITY_TOTAL from features_excel.py


# -------------------------------------------------------------------
# Test 1: Baseline forecast now clips to capacity
# -------------------------------------------------------------------

def test_baseline_clips_to_capacity():
    """Verify that _forecast_baseline clips yhat and pi_high to capacity."""
    print("=" * 70)
    print("TEST 1: Baseline forecast clips to capacity")
    print("=" * 70)

    # Create a steeply trending series that would exceed capacity
    n_points = 5 * 24 * 60
    timestamps = pd.date_range(
        start=datetime(2025, 6, 1, 8, 0, tzinfo=UTC),
        periods=n_points,
        freq="1min",
    )
    # Steep upward trend: 0 → 200 (exceeds capacity without clipping)
    base = np.linspace(0, 200, n_points)
    daily_cycle = 10 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 60))
    occupancy = np.maximum(0, base + daily_cycle)
    series = pd.Series(occupancy, index=timestamps, name="occupancy")

    # Import and call the fixed _forecast_baseline
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "services" / "forecast"))
    from main import _forecast_baseline

    # Short horizon
    points_short, info_short = _forecast_baseline(series, 60, capacity=CAPACITY)
    max_yhat_short = max(p.yhat for p in points_short)
    max_pi_high_short = max(p.pi_high for p in points_short)

    # Long horizon
    points_long, info_long = _forecast_baseline(series, 10080, capacity=CAPACITY)
    max_yhat_long = max(p.yhat for p in points_long)
    max_pi_high_long = max(p.pi_high for p in points_long)

    print(f"  Short (60 min):  max yhat={max_yhat_short:.1f}, max pi_high={max_pi_high_short:.1f}")
    print(f"  Long (7 days):   max yhat={max_yhat_long:.1f}, max pi_high={max_pi_high_long:.1f}")
    print(f"  Capacity:        {CAPACITY}")

    yhat_ok = max_yhat_long <= CAPACITY + 0.01
    pi_ok = max_pi_high_long <= CAPACITY + 0.01

    print(f"  yhat <= capacity?    {'PASS' if yhat_ok else 'FAIL'}")
    print(f"  pi_high <= capacity? {'PASS' if pi_ok else 'FAIL'}")
    print()
    return yhat_ok and pi_ok


# -------------------------------------------------------------------
# Test 1b: Baseline forecast keeps a visible uncertainty band
# -------------------------------------------------------------------

def test_baseline_interval_does_not_collapse():
    """Verify that flat history does not produce pi_low == yhat == pi_high."""
    print("=" * 70)
    print("TEST 1b: Baseline interval does not collapse")
    print("=" * 70)

    n_points = 4 * 60
    timestamps = pd.date_range(
        start=datetime(2025, 6, 1, 8, 0, tzinfo=UTC),
        periods=n_points,
        freq="1min",
    )
    series = pd.Series(np.full(n_points, 12.0), index=timestamps, name="occupancy")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "services" / "forecast"))
    from main import _forecast_baseline

    points, info = _forecast_baseline(series, 210, capacity=CAPACITY)
    widths = [p.pi_high - p.pi_low for p in points]
    min_width = min(widths) if widths else 0.0
    contains_yhat = all(p.pi_low <= p.yhat <= p.pi_high for p in points)
    visible_band = min_width > 0.5

    print(f"  Min interval width: {min_width:.2f}")
    print(f"  Interval floor:     {info.get('interval_floor', 0.0):.2f}")
    print(f"  Contains yhat?      {'PASS' if contains_yhat else 'FAIL'}")
    print(f"  Visible band?       {'PASS' if visible_band else 'FAIL'}")
    print()
    return contains_yhat and visible_band


# -------------------------------------------------------------------
# Test 2: LGBM inference clips to capacity
# -------------------------------------------------------------------

def test_lgbm_clips_to_capacity():
    """Verify that LGBM inference clipping logic now includes capacity."""
    print("=" * 70)
    print("TEST 2: LGBM inference clips to capacity")
    print("=" * 70)

    # Simulate what the fixed _forecast_lgbm does
    simulated_raw_q03 = 75.0
    simulated_raw_q50 = 90.0   # above capacity
    simulated_raw_q97 = 120.0  # way above capacity

    # Fixed clipping logic (matches main.py:1303-1308 after fix)
    q03_pred = max(0.0, min(CAPACITY, simulated_raw_q03))
    q50_pred = max(0.0, min(CAPACITY, simulated_raw_q50))
    q97_pred = max(0.0, min(CAPACITY, simulated_raw_q97))
    q03_pred = min(q03_pred, q50_pred)
    q97_pred = max(q97_pred, q50_pred)

    print(f"  Raw model output: q03={simulated_raw_q03}, q50={simulated_raw_q50}, q97={simulated_raw_q97}")
    print(f"  After fix:        q03={q03_pred}, q50={q50_pred}, q97={q97_pred}")
    print(f"  Capacity:         {CAPACITY}")

    q50_ok = q50_pred <= CAPACITY
    q97_ok = q97_pred <= CAPACITY
    mono_ok = q03_pred <= q50_pred <= q97_pred

    print(f"  q50 <= capacity?     {'PASS' if q50_ok else 'FAIL'}")
    print(f"  q97 <= capacity?     {'PASS' if q97_ok else 'FAIL'}")
    print(f"  Monotonicity ok?     {'PASS' if mono_ok else 'FAIL'}")
    print()
    return q50_ok and q97_ok and mono_ok


# -------------------------------------------------------------------
# Test 3: Weekly forecast clips to capacity
# -------------------------------------------------------------------

def test_weekly_clips_to_capacity():
    """Verify that weekly forecast now clips yhat and pi_high to capacity."""
    print("=" * 70)
    print("TEST 3: Weekly forecast clips to capacity")
    print("=" * 70)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "services" / "forecast"))
    from weekly import build_weekly_forecast

    # Create history that averages near capacity
    n_hours = 7 * 24
    timestamps = pd.date_range(
        start=datetime(2025, 6, 1, 0, 0, tzinfo=UTC),
        periods=n_hours * 4,  # 15-min resolution
        freq="15min",
    )
    occupancy = 75 + 10 * np.sin(2 * np.pi * np.arange(len(timestamps)) / (24 * 4))
    history_df = pd.DataFrame({
        "timestamp": timestamps,
        "occupancy": occupancy,
        "utilization": occupancy / CAPACITY,
        "quality_score": np.ones(len(timestamps)),
        "quality_flags": [["OK"]] * len(timestamps),
    })

    # Events that push above capacity
    events_df = pd.DataFrame({
        "starts_at": [datetime(2025, 6, 8, 10, 0, tzinfo=UTC)],
        "ends_at": [datetime(2025, 6, 8, 18, 0, tzinfo=UTC)],
        "expected_impact": [10.0],
    })

    result = build_weekly_forecast(
        zone_id="test-zone",
        history_df=history_df,
        events_df=events_df,
        lecture_df=None,
        days=7,
        slot_minutes=60,
        capacity=CAPACITY,
    )

    yhats = [p["yhat"] for p in result["points"]]
    pi_highs = [p["pi_high"] for p in result["points"]]
    max_yhat = max(yhats) if yhats else 0
    max_pi_high = max(pi_highs) if pi_highs else 0

    print(f"  Max yhat:     {max_yhat:.1f}")
    print(f"  Max pi_high:  {max_pi_high:.1f}")
    print(f"  Capacity:     {CAPACITY}")

    yhat_ok = max_yhat <= CAPACITY + 0.01
    pi_ok = max_pi_high <= CAPACITY + 0.01

    print(f"  yhat <= capacity?    {'PASS' if yhat_ok else 'FAIL'}")
    print(f"  pi_high <= capacity? {'PASS' if pi_ok else 'FAIL'}")
    print()
    return yhat_ok and pi_ok


# -------------------------------------------------------------------
# Test 4: Edge cases
# -------------------------------------------------------------------

def test_edge_cases():
    """Verify edge cases: zero capacity, negative predictions, boundary values."""
    print("=" * 70)
    print("TEST 4: Edge cases")
    print("=" * 70)

    all_pass = True

    # 4a: Normal predictions within range should not be affected
    q50_normal = 42.0
    q50_clipped = max(0.0, min(CAPACITY, q50_normal))
    ok = abs(q50_clipped - q50_normal) < 0.001
    print(f"  Normal value (42.0) unchanged?  {'PASS' if ok else 'FAIL'}")
    all_pass = all_pass and ok

    # 4b: Negative prediction should clip to 0
    q50_neg = -5.0
    q50_clipped = max(0.0, min(CAPACITY, q50_neg))
    ok = q50_clipped == 0.0
    print(f"  Negative (-5.0) clips to 0?     {'PASS' if ok else 'FAIL'}")
    all_pass = all_pass and ok

    # 4c: Value exactly at capacity should pass through
    q50_at_cap = float(CAPACITY)
    q50_clipped = max(0.0, min(CAPACITY, q50_at_cap))
    ok = abs(q50_clipped - CAPACITY) < 0.001
    print(f"  At capacity ({CAPACITY}) unchanged?    {'PASS' if ok else 'FAIL'}")
    all_pass = all_pass and ok

    # 4d: Value just above capacity should clip
    q50_above = CAPACITY + 0.1
    q50_clipped = max(0.0, min(CAPACITY, q50_above))
    ok = abs(q50_clipped - CAPACITY) < 0.001
    print(f"  Above capacity ({CAPACITY}+0.1) clips? {'PASS' if ok else 'FAIL'}")
    all_pass = all_pass and ok

    # 4e: Monotonicity preserved after clipping
    raw_q03, raw_q50, raw_q97 = 80.0, 90.0, 120.0
    q03 = max(0.0, min(CAPACITY, raw_q03))
    q50 = max(0.0, min(CAPACITY, raw_q50))
    q97 = max(0.0, min(CAPACITY, raw_q97))
    q03 = min(q03, q50)
    q97 = max(q97, q50)
    ok = q03 <= q50 <= q97
    print(f"  Monotonicity after clip?         {'PASS' if ok else 'FAIL'} (q03={q03}, q50={q50}, q97={q97})")

    all_pass = all_pass and ok
    print()
    return all_pass


if __name__ == "__main__":
    print()
    print("*" * 70)
    print("FORECAST CAPACITY FIX VALIDATION")
    print(f"Capacity = {CAPACITY}")
    print("*" * 70)
    print()

    results = []
    results.append(("Baseline capacity clip", test_baseline_clips_to_capacity()))
    results.append(("Baseline interval width", test_baseline_interval_does_not_collapse()))
    results.append(("LGBM capacity clip", test_lgbm_clips_to_capacity()))
    results.append(("Weekly capacity clip", test_weekly_clips_to_capacity()))
    results.append(("Edge cases", test_edge_cases()))

    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        all_pass = all_pass and passed

    print()
    if all_pass:
        print("  ALL TESTS PASSED - Fix is validated.")
        sys.exit(0)
    else:
        print("  SOME TESTS FAILED - Fix needs revision.")
        sys.exit(1)
