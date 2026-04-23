# Scientific Training and Promotion Gate

This document defines the implemented scientific workflow for forecast model training and promotion.

## Baseline definition

The operational baseline stays active and is used in every benchmark:

- `FORECAST_MODEL_BACKEND=baseline` by default
- Forecast formula:
  - `yhat = max(0, 0.6 * seasonal_base + 0.4 * regression_pred)`
  - `seasonal_base`: lag-based seasonal naive component
  - `regression_pred`: linear regression over trend and cyclical time features
- Prediction intervals: residual quantiles `q10/q90`

Implementation: `services/forecast/main.py`.

## Scientific evaluation workflow

Implemented endpoint: `POST /v1/train/evaluate`.

1. Quality filtering
   - Keep rows with `quality_score >= min_quality_score`
   - Exclude rows with configured hard flags (default: `TRACK_ERROR`, `SERIALIZATION_ERROR`, `BACKLOG_OVERFLOW`, `ZONE_MISSING`)

2. Rolling-origin backtesting
   - Default: `6` folds
   - Per fold: `30d` train, `7d` validation, `7d` test, `60m` gap
   - No random shuffle; strict temporal ordering

3. Benchmark-first evaluation
   - Baseline always evaluated
   - Challengers: `tf_mlp`, `quantile_gbdt`, optional `sarimax` (if statsmodels available)

4. Metrics
   - Point metrics: MAE (primary), RMSE, MASE
   - Probabilistic metrics: Pinball (`q10`, `q50`, `q90`) and 90% interval coverage
   - Segment metrics per horizon under `models.<model>.horizons.<h>.segments.*`:
     - `lecture_active` (`lecture_count_now > 0`)
     - `heavy_effect` (`lecture_heavy_now + lecture_heavy_post_60m > 0`)
     - `lecture_transition_start` (`lecture_starts_next_60m > 0`)
     - `lecture_transition_end` (`lecture_ends_next_60m > 0`)
   - Segment guardrail: if `n < 30` (configurable via `segment_min_samples` / `SCIENTIFIC_EVAL_SEGMENT_MIN_SAMPLES`), segment is flagged `*_insufficient_samples=true` and metrics are not evaluated
   - Composite score:
     - `Score = 0.50*norm(MAE60) + 0.20*norm(MASE) + 0.20*norm(Pinball) + 0.10*CoveragePenalty`

5. Statistical significance
   - Diebold-Mariano test vs baseline (one-sided: candidate better than baseline)

6. Promotion gate
   - Improvement vs baseline MAE on primary horizon >= `8%`
   - No long-horizon degradation > `2%`
   - Coverage in `[85%, 95%]`
   - DM significance (`p < 0.05`, one-sided)

7. Lecture impact transparency
   - Report includes `lecture_impact_summary`:
     - share of minutes with heavy-module effect
     - mean/quantiles of `lecture_net_pull`
     - feature availability rate
   - Request toggle: `include_lecture_impact` (default `true`)

Evaluation artifacts are stored under `services/forecast/models/reports/<zone_id>/`.

## Nightly evaluate + ablation service

Implemented service: `services/forecast-trainer` (default port `8013`).

- Trigger model evaluation nightly in UTC (`FORECAST_TRAINER_NIGHTLY_UTC`, default `02:15`)
- Run paired evaluation:
  - with lecture impact (`include_lecture_impact=true`)
  - without lecture impact (`include_lecture_impact=false`)
- Persist run IDs and primary horizon deltas:
  - `mae_gain_primary_horizon`
  - `pinball_gain_primary_horizon`
  - `coverage_delta_primary_horizon`

Internal endpoints:

- `GET /health`
- `GET /status`
- `POST /run-once`

CLI helpers:

- `scripts/train/run_nightly_eval_once.sh`
- `scripts/train/promote_latest_validated.sh`
- `scripts/train/switch_backend_tf.sh`

Operating mode:

- Runtime forecast backend stays `baseline` by default.
- Promotion remains manual via `POST /v1/train/promote`.

## Promotion workflow

Implemented endpoint: `POST /v1/train/promote`.

- Input: `zone_id`, `run_id`
- Promotion is allowed only if evaluated run passed scientific gate
- Runtime deployment currently supports promoting `tf_mlp` bundles
- Model metadata is enriched with:
  - `promoted_by_run_id`
  - `test_status=scientifically_validated`
  - `scientific_validation` block

## Forecast response enrichment

When a promoted `tf_mlp` model has validation metadata:

- `model_version` includes run/test marker
- `evidence.model` includes `run_id`, `test_status`, and validation metadata

## Sources

1. TimeSeriesSplit (rolling time-series CV)
   - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
2. Proper scoring rules / probabilistic evaluation
   - Gneiting & Raftery (2007): https://academic.oup.com/jrsssb/article/69/2/243/7109375
3. Forecast benchmark evidence (M4)
   - https://www.sciencedirect.com/science/article/pii/S0169207018300785
4. Probabilistic competition context (M5)
   - https://www.sciencedirect.com/science/article/pii/S0169207019301128
5. Occupancy forecasting survey context
   - https://www.mdpi.com/2076-3417/14/1/142
