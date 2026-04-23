CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS zones (
  zone_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  capacity INTEGER NOT NULL CHECK (capacity > 0),
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS counts (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL,
  zone_id TEXT NOT NULL REFERENCES zones(zone_id) ON DELETE CASCADE,
  occupancy INTEGER NOT NULL CHECK (occupancy >= 0),
  utilization DOUBLE PRECISION NOT NULL CHECK (utilization >= 0),
  source TEXT NOT NULL,
  quality_score DOUBLE PRECISION NOT NULL DEFAULT 1.0,
  quality_flags JSONB NOT NULL DEFAULT '[]'::jsonb,
  evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_counts_zone_ts ON counts (zone_id, ts DESC);
SELECT create_hypertable('counts', 'ts', if_not_exists => TRUE, migrate_data => TRUE);

CREATE MATERIALIZED VIEW IF NOT EXISTS counts_1m
WITH (timescaledb.continuous) AS
SELECT
  zone_id,
  time_bucket('1 minute', ts) AS bucket,
  AVG(occupancy) AS occupancy_avg,
  AVG(utilization) AS utilization_avg,
  AVG(quality_score) AS quality_avg
FROM counts
GROUP BY zone_id, bucket
WITH NO DATA;

CREATE MATERIALIZED VIEW IF NOT EXISTS counts_5m
WITH (timescaledb.continuous) AS
SELECT
  zone_id,
  time_bucket('5 minute', ts) AS bucket,
  AVG(occupancy) AS occupancy_avg,
  AVG(utilization) AS utilization_avg,
  AVG(quality_score) AS quality_avg
FROM counts
GROUP BY zone_id, bucket
WITH NO DATA;

CREATE TABLE IF NOT EXISTS calendar_events (
  event_id TEXT PRIMARY KEY,
  zone_id TEXT REFERENCES zones(zone_id) ON DELETE SET NULL,
  title TEXT NOT NULL,
  category TEXT,
  starts_at TIMESTAMPTZ NOT NULL,
  ends_at TIMESTAMPTZ NOT NULL,
  expected_impact DOUBLE PRECISION,
  source TEXT NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_calendar_events_time ON calendar_events (starts_at, ends_at);

CREATE TABLE IF NOT EXISTS forecasts (
  forecast_id TEXT PRIMARY KEY,
  zone_id TEXT NOT NULL REFERENCES zones(zone_id) ON DELETE CASCADE,
  horizon INTEGER NOT NULL,
  generated_at TIMESTAMPTZ NOT NULL,
  model_version TEXT NOT NULL,
  payload JSONB NOT NULL,
  evidence JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS explanations (
  explanation_id TEXT PRIMARY KEY,
  zone_id TEXT NOT NULL REFERENCES zones(zone_id) ON DELETE CASCADE,
  horizon INTEGER NOT NULL,
  summary TEXT NOT NULL,
  payload JSONB NOT NULL,
  evidence JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS recommendations (
  recommendation_id TEXT PRIMARY KEY,
  zone_id TEXT NOT NULL REFERENCES zones(zone_id) ON DELETE CASCADE,
  horizon INTEGER NOT NULL,
  summary TEXT NOT NULL,
  payload JSONB NOT NULL,
  evidence JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS scenario_runs (
  scenario_id TEXT PRIMARY KEY,
  zone_id TEXT NOT NULL REFERENCES zones(zone_id) ON DELETE CASCADE,
  horizon INTEGER NOT NULL,
  persist BOOLEAN NOT NULL DEFAULT FALSE,
  input_payload JSONB NOT NULL,
  result_payload JSONB NOT NULL,
  evidence JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
