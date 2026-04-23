CREATE TABLE IF NOT EXISTS lecture_activity (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL,
  zone_id TEXT NOT NULL REFERENCES zones(zone_id) ON DELETE CASCADE,
  active_lectures INTEGER NOT NULL DEFAULT 0 CHECK (active_lectures >= 0),
  active_courses INTEGER NOT NULL DEFAULT 0 CHECK (active_courses >= 0),
  starts_next_60m INTEGER NOT NULL DEFAULT 0 CHECK (starts_next_60m >= 0),
  ends_next_60m INTEGER NOT NULL DEFAULT 0 CHECK (ends_next_60m >= 0),
  source TEXT NOT NULL,
  quality_score DOUBLE PRECISION NOT NULL DEFAULT 1.0 CHECK (quality_score >= 0 AND quality_score <= 1),
  quality_flags JSONB NOT NULL DEFAULT '[]'::jsonb,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (zone_id, ts)
);

CREATE INDEX IF NOT EXISTS idx_lecture_activity_zone_ts
  ON lecture_activity (zone_id, ts DESC);
