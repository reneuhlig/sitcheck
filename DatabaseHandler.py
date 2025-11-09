from datetime import datetime
from typing import Dict, Any, Optional, List
import pg8000
import json
import logging


class DatabaseHandler:
    """PostgreSQL Datenbankoperationen für Live-Personenerkennung"""
    
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
        """Verbindet zur PostgreSQL Datenbank"""
        try:
            self.connection = pg8000.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                timeout=10
            )
            self.connection.autocommit = True
            print(f"Erfolgreich mit PostgreSQL verbunden ({self.host}:{self.port})")
            return True
        except pg8000.dbapi.InterfaceError as e:
            print(f"Fehler bei Datenbankverbindung: {e}")
            return False
    
    def create_tables(self) -> bool:
        """Erstellt die benötigten Tabellen"""
        if not self.connection:
            return False
            
        cursor = self.connection.cursor()
        
        # Tabelle für Rohdaten (beide Ordner)
        create_detections_table = """
        CREATE TABLE IF NOT EXISTS live_detections (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            source VARCHAR(50) NOT NULL,
            persons_detected INTEGER NOT NULL,
            avg_confidence REAL,
            max_confidence REAL,
            min_confidence REAL,
            detection_data JSONB,
            processed BOOLEAN DEFAULT FALSE,
            used_in_pattern BOOLEAN DEFAULT FALSE,
            pattern_id INTEGER
        );
        """
        
        create_detections_indices = """
        CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON live_detections (timestamp);
        CREATE INDEX IF NOT EXISTS idx_detections_source ON live_detections (source);
        CREATE INDEX IF NOT EXISTS idx_detections_processed ON live_detections (processed);
        CREATE INDEX IF NOT EXISTS idx_detections_used_in_pattern ON live_detections (used_in_pattern);
        """
        
        # Tabelle für korrelierte/bereinigte Daten (alte Tabelle, bleibt erhalten)
        create_correlated_table = """
        CREATE TABLE IF NOT EXISTS correlated_persons (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            source_x_id INTEGER REFERENCES live_detections(id),
            source_y_id INTEGER REFERENCES live_detections(id),
            persons_x INTEGER NOT NULL,
            persons_y INTEGER NOT NULL,
            estimated_actual_persons INTEGER NOT NULL,
            confidence_score REAL,
            time_diff_seconds REAL,
            analysis_data JSONB
        );
        CREATE INDEX IF NOT EXISTS idx_correlated_timestamp ON correlated_persons (timestamp);
        """
        
        # Tabelle für Bewegungs-Tracking
        create_movement_table = """
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
        """
        
        # Tabelle für Raumzustand
        create_room_state_table = """
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
            # Erstelle Tabellen
            cursor.execute(create_detections_table)
            cursor.execute(create_correlated_table)
            cursor.execute(create_movement_table)
            cursor.execute(create_room_state_table)
            
            # Füge Spalten zu existierenden Tabellen hinzu falls nicht vorhanden
            # WICHTIG: VOR Index-Erstellung!
            try:
                cursor.execute("""
                    ALTER TABLE live_detections 
                    ADD COLUMN IF NOT EXISTS processed BOOLEAN DEFAULT FALSE
                """)
            except:
                pass
            
            try:
                cursor.execute("""
                    ALTER TABLE live_detections 
                    ADD COLUMN IF NOT EXISTS used_in_pattern BOOLEAN DEFAULT FALSE
                """)
            except:
                pass
            
            try:
                cursor.execute("""
                    ALTER TABLE live_detections 
                    ADD COLUMN IF NOT EXISTS pattern_id INTEGER
                """)
            except:
                pass
            
            # Erstelle Indices NACH Spalten-Erstellung
            cursor.execute(create_detections_indices)
            
            print("Datenbanktabellen erstellt/überprüft")
            return True
        except pg8000.dbapi.DatabaseError as e:
            print(f"Fehler beim Erstellen der Tabellen: {e}")
            return False
        finally:
            cursor.close()
            
    def insert_detection(self, source: str, persons_detected: int,
                        avg_confidence: float, max_confidence: float,
                        min_confidence: float, detection_data: Dict) -> Optional[int]:
        """
        Fügt eine neue Detection ein
        
        Returns:
            ID des eingefügten Datensatzes oder None bei Fehler
        """
        if not self.connection:
            return None
            
        cursor = self.connection.cursor()
        query = """
        INSERT INTO live_detections 
        (source, persons_detected, avg_confidence, max_confidence, min_confidence, detection_data)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
        """
        
        try:
            cursor.execute(query, (
                source,
                persons_detected,
                avg_confidence if avg_confidence else None,
                max_confidence if max_confidence else None,
                min_confidence if min_confidence else None,
                json.dumps(detection_data, ensure_ascii=False)
            ))
            result = cursor.fetchone()
            return result[0] if result else None
        except pg8000.dbapi.DatabaseError as e:
            self.logger.error(f"Fehler beim Einfügen der Detection: {e}")
            return None
        finally:
            cursor.close()
    
    def insert_correlated_result(self, source_x_id: int, source_y_id: int,
                                 persons_x: int, persons_y: int,
                                 estimated_actual: int, confidence: float,
                                 time_diff: float, analysis_data: Dict) -> bool:
        """Fügt ein korreliertes Ergebnis ein"""
        if not self.connection:
            return False
            
        cursor = self.connection.cursor()
        query = """
        INSERT INTO correlated_persons 
        (source_x_id, source_y_id, persons_x, persons_y, estimated_actual_persons,
         confidence_score, time_diff_seconds, analysis_data)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        try:
            cursor.execute(query, (
                source_x_id,
                source_y_id,
                persons_x,
                persons_y,
                estimated_actual,
                confidence,
                time_diff,
                json.dumps(analysis_data, ensure_ascii=False)
            ))
            return True
        except pg8000.dbapi.DatabaseError as e:
            self.logger.error(f"Fehler beim Einfügen des korrelierten Ergebnisses: {e}")
            return False
        finally:
            cursor.close()
    
    def get_detections_for_analysis(self, analysis_window: float = 30.0, 
                                     lookback_window: float = 60.0) -> List[Dict]:
        """
        Holt Detections für Analyse mit Lookback für zyklusübergreifende Muster
        
        Args:
            analysis_window: Zeitfenster für neue Detections (Sekunden)
            lookback_window: Zusätzliches Zeitfenster für alte Detections (Sekunden)
            
        Returns:
            Liste von Detections mit allen relevanten Feldern
        """
        if not self.connection:
            return []
            
        cursor = self.connection.cursor()
        query = """
        SELECT id, timestamp, source, persons_detected, 
               avg_confidence, detection_data, processed, used_in_pattern
        FROM live_detections
        WHERE (
            (processed = FALSE AND timestamp >= NOW() - make_interval(secs => %s))
            OR
            (processed = TRUE 
             AND used_in_pattern = FALSE 
             AND timestamp >= NOW() - make_interval(secs => %s))
        )
        ORDER BY timestamp ASC
        """
        
        try:
            cursor.execute(query, (analysis_window, lookback_window))
            results = []
            
            for row in cursor.fetchall():
                detection_data = row[5] if row[5] is not None else {}
                
                results.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'source': row[2],
                    'persons_detected': row[3],
                    'avg_confidence': row[4],
                    'detection_data': detection_data,
                    'processed': row[6],
                    'used_in_pattern': row[7]
                })
            return results
        except pg8000.dbapi.DatabaseError as e:
            self.logger.error(f"Fehler beim Abrufen der Detections für Analyse: {e}")
            return []
        finally:
            cursor.close()
    
    def get_unprocessed_detections(self, window_seconds: float = 5.0) -> List[Dict]:
        """
        DEPRECATED: Verwende get_detections_for_analysis() stattdessen
        Holt unverarbeitete Detections aus Zeitfenster
        """
        if not self.connection:
            return []
            
        cursor = self.connection.cursor()
        query = """
        SELECT id, timestamp, source, persons_detected, avg_confidence, detection_data
        FROM live_detections
        WHERE processed = FALSE
          AND timestamp >= NOW() - make_interval(secs => %s)
        ORDER BY timestamp ASC
        """
        
        try:
            cursor.execute(query, (window_seconds,))
            results = []
            
            for row in cursor.fetchall():
                detection_data = row[5] if row[5] is not None else {}
                
                results.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'source': row[2],
                    'persons_detected': row[3],
                    'avg_confidence': row[4],
                    'detection_data': detection_data
                })
            return results
        except pg8000.dbapi.DatabaseError as e:
            self.logger.error(f"Fehler beim Abrufen unverarbeiteter Detections: {e}")
            return []
        finally:
            cursor.close()
    
    def mark_detections_processed(self, detection_ids: List[int]) -> bool:
        """
        Markiert Detections als verarbeitet
        
        Args:
            detection_ids: Liste von Detection-IDs
            
        Returns:
            True bei Erfolg
        """
        if not self.connection or not detection_ids:
            return False
            
        cursor = self.connection.cursor()
        query = "UPDATE live_detections SET processed = TRUE WHERE id = ANY(%s)"
        
        try:
            cursor.execute(query, (detection_ids,))
            return True
        except pg8000.dbapi.DatabaseError as e:
            self.logger.error(f"Fehler beim Markieren der Detections: {e}")
            return False
        finally:
            cursor.close()
    
    def mark_detections_used_in_pattern(self, detection_ids: List[int], 
                                        pattern_id: int) -> bool:
        """
        Markiert Detections als in Muster verwendet
        
        Args:
            detection_ids: Liste von Detection-IDs
            pattern_id: ID des Movement-Eintrags
            
        Returns:
            True bei Erfolg
        """
        if not self.connection or not detection_ids:
            return False
            
        cursor = self.connection.cursor()
        query = """
        UPDATE live_detections 
        SET used_in_pattern = TRUE, 
            pattern_id = %s,
            processed = TRUE
        WHERE id = ANY(%s)
        """
        
        try:
            cursor.execute(query, (pattern_id, detection_ids))
            return True
        except pg8000.dbapi.DatabaseError as e:
            self.logger.error(f"Fehler beim Markieren als verwendet: {e}")
            return False
        finally:
            cursor.close()
    
    def insert_movement(self, movement_type: str, person_count: int,
                       confidence: float, detection_sequence: List[Dict],
                       notes: str = None) -> Optional[int]:
        """
        Fügt eine erkannte Bewegung ein
        
        Args:
            movement_type: 'entry', 'exit', 'cross_traffic', 'uncertain'
            person_count: Anzahl Personen bei dieser Bewegung
            confidence: Konfidenz der Erkennung
            detection_sequence: Liste der beteiligten Detections
            notes: Optionale Notizen
            
        Returns:
            ID des eingefügten Datensatzes oder None bei Fehler
        """
        if not self.connection:
            return None
            
        cursor = self.connection.cursor()
        query = """
        INSERT INTO movement_tracking 
        (movement_type, person_count, confidence_score, detection_sequence, notes)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """
        
        try:
            cursor.execute(query, (
                movement_type,
                person_count,
                confidence,
                json.dumps(detection_sequence, ensure_ascii=False),
                notes
            ))
            result = cursor.fetchone()
            return result[0] if result else None
        except pg8000.dbapi.DatabaseError as e:
            self.logger.error(f"Fehler beim Einfügen der Bewegung: {e}")
            return None
        finally:
            cursor.close()
    
    def insert_room_state(self, total_persons: int, change_reason: str,
                         movement_id: int = None, confidence: float = None,
                         notes: str = None) -> bool:
        """
        Fügt einen neuen Raumzustand ein
        
        Args:
            total_persons: Gesamtanzahl Personen im Raum
            change_reason: Grund der Änderung ('entry', 'exit', 'initialization', etc.)
            movement_id: Optionale Referenz zur Bewegung
            confidence: Optionale Konfidenz
            notes: Optionale Notizen
            
        Returns:
            True bei Erfolg
        """
        if not self.connection:
            return False
            
        cursor = self.connection.cursor()
        query = """
        INSERT INTO room_state 
        (total_persons, change_reason, movement_tracking_id, confidence, notes)
        VALUES (%s, %s, %s, %s, %s)
        """
        
        try:
            cursor.execute(query, (
                total_persons,
                change_reason,
                movement_id,
                confidence,
                notes
            ))
            return True
        except pg8000.dbapi.DatabaseError as e:
            self.logger.error(f"Fehler beim Einfügen des Raumzustands: {e}")
            return False
        finally:
            cursor.close()
    
    def get_latest_room_state(self) -> Optional[Dict]:
        """
        Holt den letzten Raumzustand aus der Datenbank
        
        Returns:
            Dictionary mit Raumzustand oder None
        """
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
            
            if row:
                return {
                    'total_persons': row[0],
                    'timestamp': row[1],
                    'change_reason': row[2],
                    'confidence': row[3]
                }
            return None
        except pg8000.dbapi.DatabaseError as e:
            self.logger.error(f"Fehler beim Abrufen des Raumzustands: {e}")
            return None
        finally:
            cursor.close()
    
    def get_latest_detections(self, limit: int = 10) -> List[Dict]:
        """Holt die neuesten Detections für Analyse"""
        if not self.connection:
            return []
            
        cursor = self.connection.cursor()
        query = """
        SELECT id, timestamp, source, persons_detected, 
               avg_confidence, max_confidence, min_confidence, detection_data
        FROM live_detections
        ORDER BY timestamp DESC
        LIMIT %s
        """
        
        try:
            cursor.execute(query, (limit,))
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'source': row[2],
                    'persons_detected': row[3],
                    'avg_confidence': row[4],
                    'max_confidence': row[5],
                    'min_confidence': row[6],
                    'detection_data': row[7] if row[7] is not None else {}
                })
            return results
        except pg8000.dbapi.DatabaseError as e:
            self.logger.error(f"Fehler beim Abrufen der Detections: {e}")
            return []
        finally:
            cursor.close()
    
    def get_paired_detections(self, max_time_diff_seconds: float = 5.0, 
                             limit: int = 100) -> List[Dict]:
        """
        Holt gepaarte Detections von beiden Quellen für Zeitreihenanalyse
        
        Args:
            max_time_diff_seconds: Maximaler Zeitunterschied für Paare
            limit: Maximale Anzahl zurückzugebender Paare
        """
        if not self.connection:
            return []
            
        cursor = self.connection.cursor()
        query = """
        WITH x_detections AS (
            SELECT id, timestamp, persons_detected, avg_confidence
            FROM live_detections
            WHERE source = 'input_x'
            ORDER BY timestamp DESC
            LIMIT %s
        ),
        y_detections AS (
            SELECT id, timestamp, persons_detected, avg_confidence
            FROM live_detections
            WHERE source = 'input_y'
            ORDER BY timestamp DESC
            LIMIT %s
        )
        SELECT 
            x.id as x_id, x.timestamp as x_time, x.persons_detected as x_persons, x.avg_confidence as x_conf,
            y.id as y_id, y.timestamp as y_time, y.persons_detected as y_persons, y.avg_confidence as y_conf,
            EXTRACT(EPOCH FROM (y.timestamp - x.timestamp)) as time_diff
        FROM x_detections x
        CROSS JOIN y_detections y
        WHERE ABS(EXTRACT(EPOCH FROM (y.timestamp - x.timestamp))) <= %s
        ORDER BY ABS(EXTRACT(EPOCH FROM (y.timestamp - x.timestamp))) ASC
        LIMIT %s
        """
        
        try:
            cursor.execute(query, (limit, limit, max_time_diff_seconds, limit))
            results = []
            for row in cursor.fetchall():
                results.append({
                    'x_id': row[0],
                    'x_time': row[1],
                    'x_persons': row[2],
                    'x_confidence': row[3],
                    'y_id': row[4],
                    'y_time': row[5],
                    'y_persons': row[6],
                    'y_confidence': row[7],
                    'time_diff': row[8]
                })
            return results
        except pg8000.dbapi.DatabaseError as e:
            self.logger.error(f"Fehler beim Abrufen gepaarter Detections: {e}")
            return []
        finally:
            cursor.close()
    
    def close(self):
        """Schließt die Datenbankverbindung"""
        if self.connection:
            self.connection.close()
            print("Datenbankverbindung geschlossen")