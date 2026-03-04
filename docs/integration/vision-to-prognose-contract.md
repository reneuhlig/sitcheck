# Vision -> Prognose Contract

## Scope
Direkte Kopplung von `bildauswertung` nach `prognose.counts` via Direct DB Write.

## Ziel
- Live-Telemetrie aus der Vision-Pipeline frame-nah in `counts` schreiben.
- Keine API-Änderung in `prognose`.
- Realtime- und Analytics-Dashboard bleiben getrennt.

## Verbindliche Feldzuordnung (`counts`)
- `ts`: UTC-Zeitpunkt des Vision-Frames.
- `zone_id`: initial `default-zone` (konfigurierbar via `integration.prognose_db.zone_id`).
- `occupancy`: aktueller Occupancy-Wert aus Runtime-State.
- `utilization`: `occupancy / capacity`, `capacity` aus `zones.capacity`.
- `source`: `vision-direct-db` (konfigurierbar).
- `quality_score`: aus Trackingzustand abgeleitet.
- `quality_flags`: z. B. `["OK"]`, `["TRACK_ERROR"]`, `["LOW_TRACK_CONF"]`.
- `evidence`: JSON mit `evidence_id`, `generated_at`, `time_window`, `sources`, `model`, `quality`.

## Schreibmodus
- `write_mode=frame_near`.
- Standardmäßig wird jede Iteration betrachtet, mit Schutz über `max_writes_per_second`.
- Default: `max_writes_per_second=10`.

## Integritätsregeln
- Wenn `strict_zone_check=true` und `zone_id` fehlt, wird **nicht** geschrieben (`ZONE_MISSING`).
- `ensure_zone=false` als Default (keine impliziten Schema-Nebenwirkungen).
- `ensure_zone=true` kann optional Zone automatisch anlegen (mit Default-Kapazität).

## Resilienz/Spooling
- Bei DB-Transportfehlern (`DB_DOWN`/`BACKEND_ERROR`) wird lokal gespult.
- Spool-Konfiguration:
  - `integration.prognose_db.spool.enabled`
  - `integration.prognose_db.spool.path`
  - `integration.prognose_db.spool.max_entries`
- Bei Recovery werden gepufferte Einträge batch-weise geflusht.

## Standard-Konfigurationsblock (`bildauswertung/config.yaml`)
```yaml
integration:
  prognose_db:
    enabled: false
    host: 127.0.0.1
    user: sitcheck
    password: change_me
    database: sitcheck
    port: 5432
    zone_id: default-zone
    source: vision-direct-db
    write_mode: frame_near
    max_writes_per_second: 10
    strict_zone_check: true
    ensure_zone: false
    spool:
      enabled: true
      path: ../website-dashboard/runtime/logs/prognose_counts_spool.jsonl
      max_entries: 20000
```

## Betriebsmetriken
- `last_successful_write_at`
- `current_write_rate`
- `spool_size`
- `spool_flush_status`
- `last_error_class` (`DB_DOWN`, `ZONE_MISSING`, `SERIALIZATION_ERROR`, `BACKLOG_OVERFLOW`, `BACKEND_ERROR`)

