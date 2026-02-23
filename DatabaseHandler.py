from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import logging

import pg8000


class DatabaseHandler:
    """Minimale PostgreSQL-Schicht für YOLO-Tracking-Okkupanz."""

    def __init__(self, host: str, user: str, password: str, database: str, port: int = 5432):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        self.connection = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def connect(self) -> bool:
        try:
            self.connection = pg8000.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                timeout=10,
            )
            self.connection.autocommit = True
            return True
        except Exception as exc:
            self.logger.error(f"DB Verbindung fehlgeschlagen: {exc}")
            return False

    def create_tables(self) -> bool:
        if not self.connection:
            return False

        cursor = self.connection.cursor()

        create_tracking_events = """
        CREATE TABLE IF NOT EXISTS tracking_events (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            event_type VARCHAR(20) NOT NULL,
            track_id BIGINT NOT NULL,
            confidence_score REAL,
            event_data JSONB,
            UNIQUE (event_type, track_id)
        );
        CREATE INDEX IF NOT EXISTS idx_tracking_events_timestamp ON tracking_events (timestamp);
        CREATE INDEX IF NOT EXISTS idx_tracking_events_event_type ON tracking_events (event_type);
        CREATE INDEX IF NOT EXISTS idx_tracking_events_track_id ON tracking_events (track_id);
        """

        create_movement_tracking = """
        CREATE TABLE IF NOT EXISTS movement_tracking (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            movement_type VARCHAR(20) NOT NULL,
            person_count INTEGER NOT NULL,
            confidence_score REAL,
            detection_sequence JSONB,
            notes TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_movement_timestamp ON movement_tracking (timestamp);
        CREATE INDEX IF NOT EXISTS idx_movement_type ON movement_tracking (movement_type);
        """

        create_room_state = """
        CREATE TABLE IF NOT EXISTS room_state (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_persons INTEGER NOT NULL,
            change_reason VARCHAR(50),
            movement_tracking_id INTEGER REFERENCES movement_tracking(id),
            confidence REAL,
            notes TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_room_state_timestamp ON room_state (timestamp);
        """

        try:
            cursor.execute(create_tracking_events)
            cursor.execute(create_movement_tracking)
            cursor.execute(create_room_state)
            return True
        except Exception as exc:
            self.logger.error(f"Fehler beim Erstellen der Tabellen: {exc}")
            return False
        finally:
            cursor.close()

    def insert_tracking_event(
        self,
        event_type: str,
        track_id: int,
        confidence: Optional[float] = None,
        event_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        if not self.connection:
            return None

        cursor = self.connection.cursor()
        query = """
        INSERT INTO tracking_events (event_type, track_id, confidence_score, event_data)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (event_type, track_id) DO NOTHING
        RETURNING id
        """

        try:
            cursor.execute(
                query,
                (
                    event_type,
                    track_id,
                    confidence,
                    json.dumps(event_data or {}, ensure_ascii=False),
                ),
            )
            row = cursor.fetchone()
            return row[0] if row else None
        except Exception as exc:
            self.logger.error(f"Fehler beim Einfügen tracking_event: {exc}")
            return None
        finally:
            cursor.close()

    def insert_movement(
        self,
        movement_type: str,
        person_count: int,
        confidence: float,
        detection_sequence: Dict[str, Any],
        notes: Optional[str] = None,
    ) -> Optional[int]:
        if not self.connection:
            return None

        cursor = self.connection.cursor()
        query = """
        INSERT INTO movement_tracking (movement_type, person_count, confidence_score, detection_sequence, notes)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """

        try:
            cursor.execute(
                query,
                (
                    movement_type,
                    person_count,
                    confidence,
                    json.dumps(detection_sequence or {}, ensure_ascii=False),
                    notes,
                ),
            )
            row = cursor.fetchone()
            return row[0] if row else None
        except Exception as exc:
            self.logger.error(f"Fehler beim Einfügen movement: {exc}")
            return None
        finally:
            cursor.close()

    def insert_room_state(
        self,
        total_persons: int,
        change_reason: str,
        movement_id: Optional[int] = None,
        confidence: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> bool:
        if not self.connection:
            return False

        cursor = self.connection.cursor()
        query = """
        INSERT INTO room_state (total_persons, change_reason, movement_tracking_id, confidence, notes)
        VALUES (%s, %s, %s, %s, %s)
        """

        try:
            cursor.execute(query, (total_persons, change_reason, movement_id, confidence, notes))
            return True
        except Exception as exc:
            self.logger.error(f"Fehler beim Einfügen room_state: {exc}")
            return False
        finally:
            cursor.close()

    def get_latest_room_state(self) -> Optional[Dict[str, Any]]:
        if not self.connection:
            return None

        cursor = self.connection.cursor()
        query = """
        SELECT total_persons, timestamp, change_reason, confidence
        FROM room_state
        ORDER BY timestamp DESC
        LIMIT 1
        """

        try:
            cursor.execute(query)
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "total_persons": row[0],
                "timestamp": row[1],
                "change_reason": row[2],
                "confidence": row[3],
            }
        except Exception as exc:
            self.logger.error(f"Fehler beim Lesen room_state: {exc}")
            return None
        finally:
            cursor.close()

    def get_occupancy_snapshot(self) -> Dict[str, Any]:
        """Für Webpage: aktueller Wert + letzte Änderung."""
        latest = self.get_latest_room_state()
        if not latest:
            return {
                "current_occupancy": 0,
                "updated_at": None,
                "change_reason": None,
                "confidence": None,
            }
        return {
            "current_occupancy": latest["total_persons"],
            "updated_at": latest["timestamp"],
            "change_reason": latest["change_reason"],
            "confidence": latest["confidence"],
        }

    def get_occupancy_timeseries(self, minutes: int = 120) -> List[Dict[str, Any]]:
        """Für Webpage/Analyse: Verlauf der Raumbelegung."""
        if not self.connection:
            return []

        cursor = self.connection.cursor()
        query = """
        SELECT timestamp, total_persons, change_reason, confidence
        FROM room_state
        WHERE timestamp >= NOW() - make_interval(mins => %s)
        ORDER BY timestamp ASC
        """

        try:
            cursor.execute(query, (minutes,))
            return [
                {
                    "timestamp": row[0],
                    "total_persons": row[1],
                    "change_reason": row[2],
                    "confidence": row[3],
                }
                for row in cursor.fetchall()
            ]
        except Exception as exc:
            self.logger.error(f"Fehler beim Lesen room_state Verlauf: {exc}")
            return []
        finally:
            cursor.close()

    def get_entry_summary(self, minutes: int = 120) -> Dict[str, Any]:
        """Für Business-Analyse: Anzahl bestätigter Eintritte im Zeitraum."""
        if not self.connection:
            return {"entries": 0, "from_minutes": minutes}

        cursor = self.connection.cursor()
        query = """
        SELECT COUNT(*)
        FROM tracking_events
        WHERE event_type = 'entry'
          AND timestamp >= NOW() - make_interval(mins => %s)
        """

        try:
            cursor.execute(query, (minutes,))
            row = cursor.fetchone()
            count = row[0] if row else 0
            return {
                "entries": int(count),
                "from_minutes": minutes,
                "as_of": datetime.now(),
            }
        except Exception as exc:
            self.logger.error(f"Fehler bei entry summary: {exc}")
            return {"entries": 0, "from_minutes": minutes}
        finally:
            cursor.close()

    def get_recent_entry_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Für Webpage/Debug: letzte bestätigte Entry-Events."""
        if not self.connection:
            return []

        cursor = self.connection.cursor()
        query = """
        SELECT id, timestamp, track_id, confidence_score, event_data
        FROM tracking_events
        WHERE event_type = 'entry'
        ORDER BY timestamp DESC
        LIMIT %s
        """

        try:
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            return [
                {
                    "id": row[0],
                    "timestamp": row[1],
                    "track_id": row[2],
                    "confidence": row[3],
                    "event_data": row[4] if row[4] else {},
                }
                for row in rows
            ]
        except Exception as exc:
            self.logger.error(f"Fehler beim Lesen entry events: {exc}")
            return []
        finally:
            cursor.close()

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None
