CREATE INDEX IF NOT EXISTS idx_forecasts_zone_horizon_generated_at
  ON forecasts (zone_id, horizon, generated_at DESC);
