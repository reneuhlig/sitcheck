from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
import logging
import os
import threading
import time
from typing import Any, Deque, Dict, Optional
from urllib import error as urlerror
from urllib import request as urlrequest

try:
    import pg8000
except Exception:  # pragma: no cover - resolved at runtime in start scripts
    pg8000 = None

from .evidence_builder import build_evidence
from .quality_mapping import map_tracking_quality


class ZoneMissingError(RuntimeError):
    """Raised when configured zone does not exist in prognose.zones."""


class PrognoseDbWriter:
    """Direct writer from vision runtime to prognose.counts with local spooling."""

    def __init__(
        self,
        *,
        config: Dict[str, Any],
        config_dir: str,
        component_name: str,
        logger_name: str = "sitcheck.integration.prognose_db_writer",
    ):
        self.enabled = bool(config.get("enabled", False))
        self.component_name = str(component_name)

        self.host = str(config.get("host", "127.0.0.1"))
        self.user = str(config.get("user", "sitcheck"))
        self.password = str(config.get("password", "change_me"))
        self.database = str(config.get("database", "sitcheck"))
        self.port = int(config.get("port", 5432))

        self.zone_id = str(config.get("zone_id", "default-zone"))
        self.source = str(config.get("source", "vision-direct-db"))
        self.write_mode = str(config.get("write_mode", "frame_near"))
        self.max_writes_per_second = max(0.1, float(config.get("max_writes_per_second", 10)))
        self._min_write_interval = 1.0 / self.max_writes_per_second
        self.strict_zone_check = bool(config.get("strict_zone_check", True))
        self.ensure_zone = bool(config.get("ensure_zone", False))
        self.default_zone_capacity = max(1, int(config.get("default_zone_capacity", 100)))
        self.capacity_refresh_seconds = max(5.0, float(config.get("capacity_refresh_seconds", 30)))
        self.flush_batch_size = max(1, int(config.get("flush_batch_size", 200)))
        self.api_ingest_enabled = bool(config.get("api_ingest_enabled", False))
        self.api_base_url = str(config.get("api_base_url", "http://127.0.0.1:8000")).rstrip("/")
        self.api_timeout_seconds = max(1.0, float(config.get("api_timeout_seconds", 8.0)))

        spool_cfg = dict(config.get("spool", {}) or {})
        self.spool_enabled = bool(spool_cfg.get("enabled", True))
        self.spool_path = self._resolve_path(
            str(
                spool_cfg.get(
                    "path",
                    "../website-dashboard/runtime/logs/prognose_counts_spool.jsonl",
                )
            ),
            config_dir=config_dir,
        )
        self.spool_max_entries = max(100, int(spool_cfg.get("max_entries", 20000)))

        self.log_path = self._resolve_path(
            str(config.get("log_path", "../website-dashboard/runtime/logs/integration_writer.log")),
            config_dir=config_dir,
        )
        self.logger = self._build_logger(logger_name)

        self._conn = None
        self._conn_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._spool_lock = threading.Lock()

        self._zone_capacity: Optional[int] = None
        self._last_capacity_refresh_mono = 0.0
        self._last_write_mono = 0.0

        self._recent_success_ts: Deque[float] = deque(maxlen=300)
        self._rate_window_seconds = 30.0

        self._writes_ok = 0
        self._writes_failed = 0
        self._writes_spooled = 0
        self._writes_flushed = 0
        self._writes_rate_limited = 0
        self._spool_overflow_dropped = 0

        self._last_successful_write_at: Optional[str] = None
        self._last_error_at: Optional[str] = None
        self._last_error_class: str = ""
        self._last_error_message: str = ""
        self._last_flush_at: Optional[str] = None

        self._spool_size = self._count_spool_lines() if self.spool_enabled else 0

    def write_frame(
        self,
        *,
        occupancy: int,
        tracks: list[dict[str, Any]],
        run_tracking_now: bool,
        track_ok: bool,
        track_error: str,
        model_name: str,
        model_version: str,
        frame_id: int | None = None,
        events_in_frame: Dict[str, int] | None = None,
    ) -> bool:
        if not self.enabled:
            return False

        with self._write_lock:
            now_mono = time.monotonic()
            if (now_mono - self._last_write_mono) < self._min_write_interval:
                self._writes_rate_limited += 1
                return False

            generated_at = datetime.now(timezone.utc)
            record: Dict[str, Any] | None = None

            try:
                zone_capacity = self._refresh_zone_capacity()
                if zone_capacity is None:
                    zone_capacity = self.default_zone_capacity

                occ = max(0, int(occupancy))
                utilization = max(0.0, float(occ) / float(zone_capacity))
                quality_score, quality_flags = map_tracking_quality(
                    track_ok=track_ok,
                    track_error=track_error,
                    tracks=tracks,
                    run_tracking_now=run_tracking_now,
                )
                evidence = build_evidence(
                    zone_id=self.zone_id,
                    source=self.source,
                    model_name=str(model_name or "yolo"),
                    model_version=str(model_version or "unknown"),
                    quality_score=quality_score,
                    quality_flags=quality_flags,
                    occupancy=occ,
                    utilization=utilization,
                    frame_id=frame_id,
                    track_count=len(tracks or []),
                    events_in_frame=events_in_frame or {},
                    time_window_seconds=self._min_write_interval,
                    generated_at=generated_at,
                )

                record = {
                    "ts": generated_at,
                    "zone_id": self.zone_id,
                    "occupancy": occ,
                    "utilization": utilization,
                    "source": self.source,
                    "quality_score": quality_score,
                    "quality_flags": quality_flags,
                    "evidence": evidence,
                }

                self._flush_spool(max_records=self.flush_batch_size)
                self._insert_record(record)

                self._writes_ok += 1
                self._last_successful_write_at = generated_at.isoformat()
                self._last_write_mono = now_mono
                self._record_success(now_mono)
                self._clear_error()
                return True
            except Exception as exc:
                error_class = self._classify_error(exc)
                self._writes_failed += 1
                self._set_error(error_class, str(exc))
                self._close_connection()

                # Spool only transport/backend errors; not semantic errors like missing zones.
                if record is not None and self.spool_enabled and error_class in {"DB_DOWN", "BACKEND_ERROR"}:
                    try:
                        self._append_spool(record)
                    except Exception as spool_exc:
                        self._set_error("BACKLOG_OVERFLOW", str(spool_exc))
                return False

    def get_status(self) -> Dict[str, Any]:
        with self._write_lock:
            self._trim_recent_successes()
            flush_status = "idle" if self._spool_size == 0 else "pending"
            return {
                "enabled": self.enabled,
                "api_ingest_enabled": self.api_ingest_enabled,
                "api_base_url": self.api_base_url if self.api_ingest_enabled else "",
                "component": self.component_name,
                "zone_id": self.zone_id,
                "source": self.source,
                "write_mode": self.write_mode,
                "max_writes_per_second": self.max_writes_per_second,
                "strict_zone_check": self.strict_zone_check,
                "ensure_zone": self.ensure_zone,
                "zone_capacity": self._zone_capacity,
                "last_successful_write_at": self._last_successful_write_at,
                "current_write_rate": round(
                    len(self._recent_success_ts) / max(1.0, self._rate_window_seconds),
                    3,
                ),
                "writes_ok": self._writes_ok,
                "writes_failed": self._writes_failed,
                "writes_rate_limited": self._writes_rate_limited,
                "writes_spooled": self._writes_spooled,
                "writes_flushed": self._writes_flushed,
                "spool_enabled": self.spool_enabled,
                "spool_path": self.spool_path if self.spool_enabled else "",
                "spool_size": self._spool_size,
                "spool_flush_status": flush_status,
                "spool_overflow_dropped": self._spool_overflow_dropped,
                "last_flush_at": self._last_flush_at,
                "last_error_class": self._last_error_class,
                "last_error_message": self._last_error_message,
                "last_error_at": self._last_error_at,
            }

    def close(self):
        with self._write_lock:
            self._close_connection()

    def _refresh_zone_capacity(self) -> Optional[int]:
        now_mono = time.monotonic()
        if (
            self._zone_capacity is not None
            and (now_mono - self._last_capacity_refresh_mono) <= self.capacity_refresh_seconds
        ):
            return self._zone_capacity

        if self.api_ingest_enabled:
            zones_payload = self._api_get_json("/api/v1/zones")
            zone_row = None
            if isinstance(zones_payload, list):
                for candidate in zones_payload:
                    if isinstance(candidate, dict) and str(candidate.get("zone_id", "")) == self.zone_id:
                        zone_row = candidate
                        break
            self._last_capacity_refresh_mono = now_mono
            if zone_row is None:
                self._zone_capacity = None
                if self.strict_zone_check:
                    raise ZoneMissingError(
                        f"zone_id '{self.zone_id}' is missing in prognose.zones (strict_zone_check=true)"
                    )
                return None
            self._zone_capacity = max(1, int(zone_row.get("capacity", self.default_zone_capacity)))
            return self._zone_capacity

        conn = self._ensure_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT capacity FROM zones WHERE zone_id = %s LIMIT 1", (self.zone_id,))
            row = cursor.fetchone()

            if row is None and self.ensure_zone:
                zone_meta = json.dumps({"created_by": "vision-direct-db", "component": self.component_name})
                cursor.execute(
                    """
                    INSERT INTO zones (zone_id, name, capacity, is_active, metadata)
                    VALUES (%s, %s, %s, TRUE, %s::jsonb)
                    ON CONFLICT (zone_id) DO NOTHING
                    """,
                    (
                        self.zone_id,
                        f"Vision Zone ({self.zone_id})",
                        self.default_zone_capacity,
                        zone_meta,
                    ),
                )
                cursor.execute("SELECT capacity FROM zones WHERE zone_id = %s LIMIT 1", (self.zone_id,))
                row = cursor.fetchone()

            self._last_capacity_refresh_mono = now_mono
            if row is None:
                self._zone_capacity = None
                if self.strict_zone_check:
                    raise ZoneMissingError(
                        f"zone_id '{self.zone_id}' is missing in prognose.zones (strict_zone_check=true)"
                    )
                return None

            self._zone_capacity = max(1, int(row[0]))
            return self._zone_capacity
        finally:
            cursor.close()

    def _insert_record(self, record: Dict[str, Any]):
        if self.api_ingest_enabled:
            ts = self._normalize_datetime(record.get("ts"))
            payload = {
                "points": [
                    {
                        "timestamp": ts.isoformat(),
                        "zone_id": str(record.get("zone_id", self.zone_id)),
                        "occupancy": int(record.get("occupancy", 0)),
                        "utilization": float(record.get("utilization", 0.0)),
                        "source": str(record.get("source", self.source)),
                        "quality_score": float(record.get("quality_score", 1.0)),
                        "quality_flags": list(record.get("quality_flags", []) or []),
                        "evidence": dict(record.get("evidence", {})),
                    }
                ]
            }
            self._api_post_json("/api/v1/ingest/counts", payload)
            return

        conn = self._ensure_connection()
        cursor = conn.cursor()
        try:
            ts = self._normalize_datetime(record.get("ts"))
            quality_flags_json = json.dumps(record.get("quality_flags", []), ensure_ascii=False)
            evidence_json = json.dumps(record.get("evidence", {}), ensure_ascii=False)
            cursor.execute(
                """
                INSERT INTO counts (
                    ts, zone_id, occupancy, utilization, source, quality_score, quality_flags, evidence
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                """,
                (
                    ts,
                    str(record.get("zone_id", self.zone_id)),
                    int(record.get("occupancy", 0)),
                    float(record.get("utilization", 0.0)),
                    str(record.get("source", self.source)),
                    float(record.get("quality_score", 1.0)),
                    quality_flags_json,
                    evidence_json,
                ),
            )
        finally:
            cursor.close()

    def _flush_spool(self, max_records: int):
        if not self.spool_enabled or self._spool_size <= 0:
            return

        with self._spool_lock:
            lines = self._read_spool_lines_unlocked()
            if not lines:
                self._spool_size = 0
                return

            pending = lines[:max_records]
            tail = lines[max_records:]
            failed_index: Optional[int] = None
            flushed_count = 0

            for idx, line in enumerate(pending):
                payload = self._deserialize_spool_record(line)
                if payload is None:
                    flushed_count += 1
                    continue

                try:
                    self._insert_record(payload)
                    flushed_count += 1
                    self._writes_flushed += 1
                    self._record_success(time.monotonic())
                except Exception as exc:
                    failed_index = idx
                    self._set_error(self._classify_error(exc), str(exc))
                    self._close_connection()
                    break

            if failed_index is None:
                remaining = tail
            else:
                remaining = pending[failed_index:] + tail

            self._write_spool_lines_unlocked(remaining)
            self._spool_size = len(remaining)
            if flushed_count > 0:
                self._last_flush_at = datetime.now(timezone.utc).isoformat()

    def _append_spool(self, record: Dict[str, Any]):
        if not self.spool_enabled:
            return

        line = self._serialize_spool_record(record)
        os.makedirs(os.path.dirname(self.spool_path), exist_ok=True)

        with self._spool_lock:
            lines = self._read_spool_lines_unlocked()
            lines.append(line)
            if len(lines) > self.spool_max_entries:
                overflow = len(lines) - self.spool_max_entries
                self._spool_overflow_dropped += overflow
                self._set_error("BACKLOG_OVERFLOW", f"spool overflow dropped={overflow}")
                lines = lines[overflow:]
            self._write_spool_lines_unlocked(lines)
            self._spool_size = len(lines)
            self._writes_spooled += 1

    def _build_logger(self, logger_name: str) -> logging.Logger:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(stream_handler)

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        log_path_abs = os.path.abspath(self.log_path)
        has_file_handler = any(
            isinstance(h, logging.FileHandler) and os.path.abspath(getattr(h, "baseFilename", "")) == log_path_abs
            for h in logger.handlers
        )
        if not has_file_handler:
            file_handler = logging.FileHandler(log_path_abs, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(file_handler)

        return logger

    def _api_get_json(self, path: str) -> Any:
        if not self.api_base_url:
            raise RuntimeError("api_base_url is empty while api_ingest_enabled=true")
        req = urlrequest.Request(
            f"{self.api_base_url}{path}",
            method="GET",
            headers={"accept": "application/json"},
        )
        try:
            with urlrequest.urlopen(req, timeout=self.api_timeout_seconds) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                return json.loads(body) if body else {}
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
            raise RuntimeError(f"api_get_failed status={exc.code} path={path} detail={detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"api_get_failed path={path} error={exc}") from exc

    def _api_post_json(self, path: str, payload: Dict[str, Any]) -> Any:
        if not self.api_base_url:
            raise RuntimeError("api_base_url is empty while api_ingest_enabled=true")
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urlrequest.Request(
            f"{self.api_base_url}{path}",
            data=encoded,
            method="POST",
            headers={
                "content-type": "application/json",
                "accept": "application/json",
            },
        )
        try:
            with urlrequest.urlopen(req, timeout=self.api_timeout_seconds) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                return json.loads(body) if body else {}
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
            raise RuntimeError(f"api_post_failed status={exc.code} path={path} detail={detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"api_post_failed path={path} error={exc}") from exc

    def _ensure_connection(self):
        if self.api_ingest_enabled:
            raise RuntimeError("_ensure_connection called in api_ingest mode")
        if pg8000 is None:
            raise RuntimeError("pg8000 is not installed; install dependency 'pg8000' first")
        with self._conn_lock:
            if self._conn is not None:
                return self._conn
            self._conn = pg8000.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                timeout=8,
            )
            self._conn.autocommit = True
            return self._conn

    def _close_connection(self):
        with self._conn_lock:
            if self._conn is None:
                return
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def _record_success(self, ts_mono: float):
        self._recent_success_ts.append(ts_mono)
        self._trim_recent_successes()

    def _trim_recent_successes(self):
        threshold = time.monotonic() - self._rate_window_seconds
        while self._recent_success_ts and self._recent_success_ts[0] < threshold:
            self._recent_success_ts.popleft()

    def _set_error(self, error_class: str, message: str):
        self._last_error_class = error_class
        self._last_error_message = str(message)
        self._last_error_at = datetime.now(timezone.utc).isoformat()
        self.logger.warning("%s | %s", error_class, message)

    def _clear_error(self):
        self._last_error_class = ""
        self._last_error_message = ""
        self._last_error_at = None

    @staticmethod
    def _classify_error(exc: Exception) -> str:
        if isinstance(exc, ZoneMissingError):
            return "ZONE_MISSING"

        raw = str(exc).lower()
        if any(token in raw for token in ("connection refused", "could not connect", "timeout", "network", "closed")):
            return "DB_DOWN"
        if any(token in raw for token in ("json", "serial", "evidence")):
            return "SERIALIZATION_ERROR"
        return "BACKEND_ERROR"

    @staticmethod
    def _normalize_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        if isinstance(value, str):
            normalized = value.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        return datetime.now(timezone.utc)

    @staticmethod
    def _resolve_path(path_value: str, *, config_dir: str) -> str:
        if os.path.isabs(path_value):
            return path_value
        return os.path.abspath(os.path.join(config_dir, path_value))

    def _serialize_spool_record(self, record: Dict[str, Any]) -> str:
        serializable = dict(record)
        serializable["ts"] = self._normalize_datetime(serializable.get("ts")).isoformat()
        return json.dumps(serializable, ensure_ascii=False)

    def _deserialize_spool_record(self, line: str) -> Optional[Dict[str, Any]]:
        stripped = line.strip()
        if not stripped:
            return None
        try:
            payload = json.loads(stripped)
            payload["ts"] = self._normalize_datetime(payload.get("ts"))
            return payload
        except Exception:
            return None

    def _count_spool_lines(self) -> int:
        if not os.path.exists(self.spool_path):
            return 0
        count = 0
        with open(self.spool_path, "r", encoding="utf-8") as handle:
            for _ in handle:
                count += 1
        return count

    def _read_spool_lines_unlocked(self) -> list[str]:
        if not os.path.exists(self.spool_path):
            return []
        with open(self.spool_path, "r", encoding="utf-8") as handle:
            return [line.rstrip("\n") for line in handle if line.strip()]

    def _write_spool_lines_unlocked(self, lines: list[str]):
        os.makedirs(os.path.dirname(self.spool_path), exist_ok=True)
        temp_path = f"{self.spool_path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(f"{line}\n")
        os.replace(temp_path, self.spool_path)
