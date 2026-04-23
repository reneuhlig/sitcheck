CREATE TABLE IF NOT EXISTS reference_objects (
  reference_id TEXT PRIMARY KEY,
  zone_id TEXT REFERENCES zones(zone_id) ON DELETE SET NULL,
  reference_type TEXT NOT NULL,
  source_type TEXT NOT NULL,
  label TEXT NOT NULL,
  uri_or_path TEXT,
  checksum TEXT,
  imported_at TIMESTAMPTZ,
  time_from TIMESTAMPTZ,
  time_to TIMESTAMPTZ,
  row_count INTEGER,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reference_objects_zone_type_created_at
  ON reference_objects (zone_id, reference_type, created_at DESC);

CREATE TABLE IF NOT EXISTS model_runs (
  model_run_id TEXT PRIMARY KEY,
  zone_id TEXT NOT NULL REFERENCES zones(zone_id) ON DELETE CASCADE,
  product TEXT NOT NULL DEFAULT 'short_term',
  horizon INTEGER,
  model_backend TEXT NOT NULL,
  model_version TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'trained',
  scientific_status TEXT NOT NULL DEFAULT 'training_only',
  include_lecture_impact BOOLEAN NOT NULL DEFAULT TRUE,
  feature_set_version TEXT NOT NULL,
  history_from TIMESTAMPTZ,
  history_to TIMESTAMPTZ,
  raw_rows INTEGER,
  train_rows INTEGER,
  val_rows INTEGER,
  test_rows INTEGER,
  evaluation_run_id TEXT,
  promoted BOOLEAN NOT NULL DEFAULT FALSE,
  promoted_at TIMESTAMPTZ,
  promotion_source TEXT,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  lineage JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_model_runs_zone_product_created_at
  ON model_runs (zone_id, product, created_at DESC);

CREATE TABLE IF NOT EXISTS model_run_references (
  id BIGSERIAL PRIMARY KEY,
  model_run_id TEXT NOT NULL REFERENCES model_runs(model_run_id) ON DELETE CASCADE,
  reference_id TEXT NOT NULL REFERENCES reference_objects(reference_id) ON DELETE CASCADE,
  relation_type TEXT NOT NULL DEFAULT 'training_data',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_model_run_references_model_run
  ON model_run_references (model_run_id, relation_type, created_at DESC);

ALTER TABLE forecasts ADD COLUMN IF NOT EXISTS product TEXT NOT NULL DEFAULT 'short_term';
ALTER TABLE forecasts ADD COLUMN IF NOT EXISTS slot_minutes INTEGER;
ALTER TABLE forecasts ADD COLUMN IF NOT EXISTS days INTEGER;
ALTER TABLE forecasts ADD COLUMN IF NOT EXISTS model_run_id TEXT;

CREATE INDEX IF NOT EXISTS idx_forecasts_zone_product_generated_at
  ON forecasts (zone_id, product, generated_at DESC);
