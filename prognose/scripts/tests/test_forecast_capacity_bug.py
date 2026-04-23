"""Reproduction test: forecast predictions exceeding physical capacity.

Tests three forecast paths for missing upper-bound clipping:
1. LGBM inference path (main.py:_forecast_lgbm)
2. Baseline forecast (main.py:_forecast_baseline)
3. Weekly forecast (weekly.py:build_weekly_forecast)

Expected: All forecast values must stay within [0, CAPACITY].
Actual (before fix): Values can exceed CAPACITY without bound.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

UTC = timezone.utc

# -------------------------------------------------------------------
# Test 1: Baseline forecast with upward-trending data -> unbounded extrapolation
# -------------------------------------------------------------------

def test_baseline_unbounded_extrapolation():
    """Demonstrate that _forecast_baseline produces values > capacity for long horizons."""

    # Simulate 3 days of data with an upward trend (e.g., growing occupancy)
    n_points = 3 * 24 * 60  # 3 days, 1-min resolution
    timestamps = pd.date_range(
        start=datetime(2025, 6, 1, 8, 0, tzinfo=UTC),
        periods=n_points,
        freq="1min",
    )
    # Trending data: starts at 10, ends at ~50 with daily cycle
    base_trend = np.linspace(10, 50, n_points)
    daily_cycle = 15 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 60) - np.pi / 3)
    occupancy = np.maximum(0, base_trend + daily_cycle + np.random.normal(0, 2, n_points))

    series = pd.Series(occupancy, index=timestamps, name="occupancy")

    CAPACITY = 84

    # Test short horizon (60 min) - should be reasonable
    horizon_short = 60
    step_minutes_short = 1
    steps_short = max(1, horizon_short // step_minutes_short)
    values = series.values.astype(float)
    n = len(values)

    from sklearn.linear_model import LinearRegression
    x = np.arange(n)
    minute_of_day = np.array([ts.hour * 60 + ts.minute for ts in series.index])
    sin_t = np.sin(2 * np.pi * minute_of_day / 1440)
    cos_t = np.cos(2 * np.pi * minute_of_day / 1440)
    X = np.column_stack([x, sin_t, cos_t])
    model = LinearRegression()
    model.fit(X, values)

    # Short horizon
    future_x_short = np.arange(n, n + steps_short)
    last_ts = series.index[-1]
    future_idx_short = [last_ts + timedelta(minutes=i) for i in range(1, steps_short + 1)]
    future_minute_short = np.array([ts.hour * 60 + ts.minute for ts in future_idx_short])
    future_X_short = np.column_stack([
        future_x_short,
        np.sin(2 * np.pi * future_minute_short / 1440),
        np.cos(2 * np.pi * future_minute_short / 1440),
    ])
    regression_pred_short = model.predict(future_X_short)
    max_short = float(regression_pred_short.max())

    # Long horizon (7 days = 10080 min)
    horizon_long = 10080
    step_minutes_long = 60
    steps_long = max(1, horizon_long // step_minutes_long)

    future_x_long = np.arange(n, n + steps_long)
    future_idx_long = [last_ts + timedelta(minutes=step_minutes_long * i) for i in range(1, steps_long + 1)]
    future_minute_long = np.array([ts.hour * 60 + ts.minute for ts in future_idx_long])
    future_X_long = np.column_stack([
        future_x_long,
        np.sin(2 * np.pi * future_minute_long / 1440),
        np.cos(2 * np.pi * future_minute_long / 1440),
    ])
    regression_pred_long = model.predict(future_X_long)
    max_long = float(regression_pred_long.max())

    # Very long horizon (14 days = 20160 min)
    horizon_vlong = 20160
    steps_vlong = max(1, horizon_vlong // step_minutes_long)
    future_x_vlong = np.arange(n, n + steps_vlong)
    future_idx_vlong = [last_ts + timedelta(minutes=step_minutes_long * i) for i in range(1, steps_vlong + 1)]
    future_minute_vlong = np.array([ts.hour * 60 + ts.minute for ts in future_idx_vlong])
    future_X_vlong = np.column_stack([
        future_x_vlong,
        np.sin(2 * np.pi * future_minute_vlong / 1440),
        np.cos(2 * np.pi * future_minute_vlong / 1440),
    ])
    regression_pred_vlong = model.predict(future_X_vlong)
    max_vlong = float(regression_pred_vlong.max())

    print("=" * 70)
    print("TEST 1: Baseline LinearRegression Extrapolation")
    print("=" * 70)
    print(f"  Training data range: [{float(occupancy.min()):.1f}, {float(occupancy.max()):.1f}]")
    print(f"  Physical capacity:   {CAPACITY}")
    print(f"  Regression slope:    {model.coef_[0]:.6f} per minute")
    print()
    print(f"  Short horizon (60 min):    max regression = {max_short:.1f}")
    print(f"  Long horizon (7 days):     max regression = {max_long:.1f}")
    print(f"  Very long horizon (14 d):  max regression = {max_vlong:.1f}")
    print()

    exceeds_short = max_short > CAPACITY
    exceeds_long = max_long > CAPACITY
    exceeds_vlong = max_vlong > CAPACITY

    print(f"  Exceeds capacity (60 min)?   {'YES - BUG' if exceeds_short else 'no'}")
    print(f"  Exceeds capacity (7 days)?   {'YES - BUG' if exceeds_long else 'no'}")
    print(f"  Exceeds capacity (14 days)?  {'YES - BUG' if exceeds_vlong else 'no'}")
    print()
    return exceeds_long or exceeds_vlong


# -------------------------------------------------------------------
# Test 2: LGBM inference missing capacity clip
# -------------------------------------------------------------------

def test_lgbm_no_upper_clip():
    """Show that LGBM inference path only clips at 0, not at capacity."""
    print("=" * 70)
    print("TEST 2: LGBM Inference - Missing Upper Capacity Clip")
    print("=" * 70)

    # Simulate what _forecast_lgbm does (main.py:1283-1292)
    # Raw model output can be anything
    simulated_raw_q03 = 75.0
    simulated_raw_q50 = 90.0   # above capacity!
    simulated_raw_q97 = 120.0  # way above capacity!

    CAPACITY = 84

    # Current inference code (main.py:1287-1292)
    q03_current = max(0.0, simulated_raw_q03)
    q50_current = max(0.0, simulated_raw_q50)
    q97_current = max(0.0, simulated_raw_q97)
    q03_current = min(q03_current, q50_current)
    q97_current = max(q97_current, q50_current)

    # What predict_gbdt does (model_gbdt.py:152-154)
    q03_correct = np.clip(simulated_raw_q03, 0, CAPACITY)
    q50_correct = np.clip(simulated_raw_q50, 0, CAPACITY)
    q97_correct = np.clip(simulated_raw_q97, 0, CAPACITY)
    q03_correct = min(q03_correct, q50_correct)
    q97_correct = max(q97_correct, q50_correct)

    print(f"  Raw model output:         q03={simulated_raw_q03}, q50={simulated_raw_q50}, q97={simulated_raw_q97}")
    print(f"  Capacity:                 {CAPACITY}")
    print()
    print(f"  Current inference result: q03={q03_current}, q50={q50_current}, q97={q97_current}")
    print(f"  Correct (with clip):      q03={q03_correct}, q50={q50_correct}, q97={q97_correct}")
    print()

    bug = q50_current > CAPACITY or q97_current > CAPACITY
    print(f"  q50 exceeds capacity?     {'YES - BUG' if q50_current > CAPACITY else 'no'} ({q50_current} vs {CAPACITY})")
    print(f"  q97 exceeds capacity?     {'YES - BUG' if q97_current > CAPACITY else 'no'} ({q97_current} vs {CAPACITY})")
    print()
    return bug


# -------------------------------------------------------------------
# Test 3: Weekly forecast missing capacity clip
# -------------------------------------------------------------------

def test_weekly_no_capacity_clip():
    """Show that weekly forecast has no upper capacity bound."""
    print("=" * 70)
    print("TEST 3: Weekly Forecast - Missing Capacity Clip")
    print("=" * 70)

    CAPACITY = 84

    # Simulate weekly forecast calculation (weekly.py:329-371)
    # High base_mean from history + event delta + lecture delta
    base_mean = 70.0  # historical average for this slot
    event_delta = 5.0 * 5.0   # event_impact_sum=5, weight=5.0 -> +25
    lecture_delta = 0.04 * 200 + 1.0 * 3 - 0.5 * 1  # net_pull + starts - ends -> +10.5

    yhat_current = max(0.0, base_mean + event_delta + lecture_delta)
    yhat_correct = min(CAPACITY, max(0.0, base_mean + event_delta + lecture_delta))

    print(f"  base_mean:       {base_mean}")
    print(f"  event_delta:     {event_delta}")
    print(f"  lecture_delta:   {lecture_delta:.1f}")
    print(f"  Capacity:        {CAPACITY}")
    print()
    print(f"  Current yhat:    {yhat_current}")
    print(f"  Correct yhat:    {yhat_correct}")
    print()

    bug = yhat_current > CAPACITY
    print(f"  Exceeds capacity? {'YES - BUG' if bug else 'no'} ({yhat_current} vs {CAPACITY})")
    print()
    return bug


# -------------------------------------------------------------------
# Test 4: Discrepancy between predict_gbdt and inference usage
# -------------------------------------------------------------------

def test_predict_gbdt_vs_inference_discrepancy():
    """Document the discrepancy between predict_gbdt (training) and inference path."""
    print("=" * 70)
    print("TEST 4: Code Discrepancy Analysis")
    print("=" * 70)
    print()
    print("  model_gbdt.py:predict_gbdt() [used in TRAINING EVALUATION]:")
    print("    q03 = bundle.model_q10.predict(X).clip(0, capacity)")
    print("    q50 = bundle.model_q50.predict(X).clip(0, capacity)")
    print("    q97 = bundle.model_q90.predict(X).clip(0, capacity)")
    print("    --> Clips to [0, capacity=84]")
    print()
    print("  main.py:_forecast_lgbm() [used in LIVE INFERENCE]:")
    print("    q03_pred = float(model_q10.predict(X_row)[0])")
    print("    q50_pred = float(model_q50.predict(X_row)[0])")
    print("    q97_pred = float(model_q90.predict(X_row)[0])")
    print("    q03_pred = max(0.0, q03_pred)  # only lower clip!")
    print("    --> NO upper capacity clip!")
    print()
    print("  main.py:_forecast_baseline() [used for LONG HORIZONS]:")
    print("    yhat = np.maximum(0.0, 0.6 * seasonal + 0.4 * regression)")
    print("    --> NO upper capacity clip!")
    print()
    print("  weekly.py:build_weekly_forecast() [used for WEEKLY OUTLOOK]:")
    print("    yhat = max(0.0, base_mean + event_delta + lecture_delta)")
    print("    --> NO upper capacity clip!")
    print()
    print("  CONCLUSION: predict_gbdt clips correctly but is ONLY used during")
    print("  training evaluation, NOT during live inference or any other path.")
    return True


if __name__ == "__main__":
    print()
    print("*" * 70)
    print("FORECAST CAPACITY BUG REPRODUCTION")
    print(f"Capacity = 84 (CAPACITY_TOTAL from features_excel.py)")
    print("*" * 70)
    print()

    bugs_found = 0

    if test_baseline_unbounded_extrapolation():
        bugs_found += 1

    if test_lgbm_no_upper_clip():
        bugs_found += 1

    if test_weekly_no_capacity_clip():
        bugs_found += 1

    test_predict_gbdt_vs_inference_discrepancy()

    print("=" * 70)
    print(f"SUMMARY: {bugs_found} bugs reproduced")
    print("=" * 70)
    if bugs_found > 0:
        print("  All three forecast paths lack upper capacity clipping.")
        print("  Long-term forecasts can grow without bound.")
        sys.exit(1)
    else:
        print("  All forecasts within capacity bounds.")
        sys.exit(0)
