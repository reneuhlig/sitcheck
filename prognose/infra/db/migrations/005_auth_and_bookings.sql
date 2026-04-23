CREATE TABLE IF NOT EXISTS users (
  user_id TEXT PRIMARY KEY,
  username TEXT NOT NULL,
  username_normalized TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  password_salt TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_username_normalized
  ON users (username_normalized);

CREATE TABLE IF NOT EXISTS user_sessions (
  session_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  token_hash TEXT NOT NULL UNIQUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at TIMESTAMPTZ NOT NULL,
  last_used_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id
  ON user_sessions (user_id, expires_at DESC);

CREATE INDEX IF NOT EXISTS idx_user_sessions_token_hash
  ON user_sessions (token_hash);

CREATE TABLE IF NOT EXISTS bookings (
  booking_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  zone_id TEXT NOT NULL REFERENCES zones(zone_id) ON DELETE CASCADE,
  starts_at TIMESTAMPTZ NOT NULL,
  ends_at TIMESTAMPTZ NOT NULL,
  party_size INTEGER NOT NULL DEFAULT 1 CHECK (party_size > 0),
  status TEXT NOT NULL DEFAULT 'confirmed',
  cancelled_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_bookings_user_id
  ON bookings (user_id, starts_at ASC, created_at ASC);

CREATE INDEX IF NOT EXISTS idx_bookings_zone_window
  ON bookings (zone_id, status, starts_at ASC, ends_at ASC);
