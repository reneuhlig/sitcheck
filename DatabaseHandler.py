from datetime import datetime
from typing import Dict, Any, Optional
import pg8000
import json
import logging


class DatabaseHandler:
    """Verbesserte PostgreSQL Datenbankoperationen für alle KI-Modelle (pg8000)"""
    
    def __init__(self, host: str, user: str, password: str, database: str, port: int = 5432):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        self.connection = None
        
        # Logging konfigurieren
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
            print(f"✓ Erfolgreich mit PostgreSQL Datenbank verbunden ({self.host}:{self.port})")
            return True
        except pg8000.dbapi.InterfaceError as e:
            print(f"✗ Fehler bei Datenbankverbindung: {e}")
            return False
    
    def create_tables(self) -> bool:
        """Erstellt die benötigten Tabellen falls sie nicht existieren"""
        if not self.connection:
            return False
            
        cursor = self.connection.cursor()
        
        create_runs_table = """
        CREATE TABLE IF NOT EXISTS ai_runs (
            run_id VARCHAR(36) PRIMARY KEY,
            model_name VARCHAR(100) NOT NULL,
            model_version VARCHAR(50),
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP,
            total_images INTEGER DEFAULT 0,
            successful_detections INTEGER DEFAULT 0,
            failed_detections INTEGER DEFAULT 0,
            avg_processing_time REAL,
            total_processing_time REAL,
            avg_cpu_usage REAL,
            max_cpu_usage REAL,
            avg_memory_usage REAL,
            max_memory_usage REAL,
            avg_gpu_usage REAL,
            max_gpu_usage REAL,
            status VARCHAR(20) DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
            error_message TEXT,
            config_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_runs_model_name ON ai_runs (model_name);
        CREATE INDEX IF NOT EXISTS idx_runs_start_time ON ai_runs (start_time);
        CREATE INDEX IF NOT EXISTS idx_runs_status ON ai_runs (status);
        """
        
        create_results_table = """
        CREATE TABLE IF NOT EXISTS detection_results (
            id SERIAL PRIMARY KEY,
            run_id VARCHAR(36) NOT NULL,
            image_path VARCHAR(500) NOT NULL,
            image_filename VARCHAR(255) NOT NULL,
            classification VARCHAR(100),
            model_output JSONB,
            confidence_scores TEXT,
            processing_time REAL NOT NULL,
            success BOOLEAN DEFAULT TRUE,
            persons_detected INTEGER DEFAULT 0,
            avg_confidence REAL,
            max_confidence REAL,
            min_confidence REAL,
            is_uncertain BOOLEAN DEFAULT FALSE,
            error_message TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES ai_runs(run_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_results_run_id ON detection_results (run_id);
        CREATE INDEX IF NOT EXISTS idx_results_classification ON detection_results (classification);
        CREATE INDEX IF NOT EXISTS idx_results_timestamp ON detection_results (timestamp);
        CREATE INDEX IF NOT EXISTS idx_results_persons_detected ON detection_results (persons_detected);
        CREATE INDEX IF NOT EXISTS idx_results_success ON detection_results (success);
        """
        
        try:
            cursor.execute(create_runs_table)
            cursor.execute(create_results_table)
            print("✓ Datenbanktabellen erstellt/überprüft")
            return True
        except pg8000.dbapi.DatabaseError as e:
            print(f"✗ Fehler beim Erstellen der Tabellen: {e}")
            return False
        finally:
            cursor.close()
            
    def insert_run(self, run_id: str, model_name: str, model_version: str = None, 
                   config: Dict = None) -> bool:
        """Fügt einen neuen Run in die Datenbank ein"""
        if not self.connection:
            return False
            
        cursor = self.connection.cursor()
        query = """
        INSERT INTO ai_runs (run_id, model_name, model_version, start_time, config_json)
        VALUES (%s, %s, %s, %s, %s)
        """
        
        try:
            model_version = str(model_version) if model_version else 'unknown'
            config_json = json.dumps(config, ensure_ascii=False) if config else None
            start_time = datetime.now()
            
            cursor.execute(query, (run_id, model_name, model_version, start_time, config_json))
            return True
        except pg8000.dbapi.DatabaseError as e:
            self.logger.error(f"Fehler beim Einfügen des Runs: {e}")
            return False
        finally:
            cursor.close()
    
    def _safe_convert_to_bool(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower().strip() in ('true', '1', 'yes', 'on')
        if isinstance(value, (int, float)):
            return value != 0
        return bool(value)
    
    def _safe_convert_to_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            result = float(value)
            if result != result:  # NaN check
                return None
            return result
        except (ValueError, TypeError):
            return None
    
    def _safe_convert_to_int_nullable(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
    
    def insert_result(self, run_id: str, image_path: str, image_filename: str,
                     classification: str, model_output: Dict, confidence_scores: str,
                     processing_time: float, success: bool = True, 
                     error_message: str = None) -> bool:
        if not self.connection:
            return False
            
        cursor = self.connection.cursor()
        query = """
        INSERT INTO detection_results 
        (run_id, image_path, image_filename, classification, model_output, 
         confidence_scores, processing_time, success, error_message, 
         persons_detected, avg_confidence, max_confidence, min_confidence, is_uncertain)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        try:
            persons_detected = self._safe_convert_to_int_nullable(
                model_output.get('persons_detected', 0) if model_output else 0
            ) or 0
            avg_confidence = self._safe_convert_to_float(model_output.get('avg_confidence')) if model_output else None
            max_confidence = self._safe_convert_to_float(model_output.get('max_confidence')) if model_output else None
            min_confidence = self._safe_convert_to_float(model_output.get('min_confidence')) if model_output else None
            is_uncertain = self._safe_convert_to_bool(model_output.get('uncertain')) if model_output else False
            success_bool = self._safe_convert_to_bool(success)
            processing_time_safe = self._safe_convert_to_float(processing_time) or 0.0
            
            params = [
                str(run_id),
                str(image_path),
                str(image_filename),
                str(classification) if classification else '',
                json.dumps(model_output, ensure_ascii=False) if model_output else None,
                str(confidence_scores) if confidence_scores else '',
                float(processing_time_safe),
                bool(success_bool),
                str(error_message) if error_message else None,
                int(persons_detected),
                float(avg_confidence) if avg_confidence is not None else None,
                float(max_confidence) if max_confidence is not None else None,
                float(min_confidence) if min_confidence is not None else None,
                bool(is_uncertain)
            ]
            
            cursor.execute(query, params)
            return True
        except pg8000.dbapi.DatabaseError as e:
            self.logger.error(f"Fehler beim Einfügen des Ergebnisses: {e}")
            return False
        finally:
            cursor.close()
    
    def update_run_completion(self, run_id: str, total_images: int, 
                            successful_detections: int, failed_detections: int,
                            avg_processing_time: float, total_processing_time: float,
                            system_stats: Dict, status: str = 'completed',
                            error_message: str = None) -> bool:
        if not self.connection:
            return False
            
        cursor = self.connection.cursor()
        query = """
        UPDATE ai_runs SET 
        end_time = %s,
        total_images = %s,
        successful_detections = %s,
        failed_detections = %s,
        avg_processing_time = %s,
        total_processing_time = %s,
        avg_cpu_usage = %s,
        max_cpu_usage = %s,
        avg_memory_usage = %s,
        max_memory_usage = %s,
        avg_gpu_usage = %s,
        max_gpu_usage = %s,
        status = %s,
        error_message = %s
        WHERE run_id = %s
        """
        
        try:
            end_time = datetime.now()
            params = (
                end_time,
                self._safe_convert_to_int_nullable(total_images) or 0,
                self._safe_convert_to_int_nullable(successful_detections) or 0,
                self._safe_convert_to_int_nullable(failed_detections) or 0,
                self._safe_convert_to_float(avg_processing_time),
                self._safe_convert_to_float(total_processing_time),
                self._safe_convert_to_float(system_stats.get('avg_cpu')),
                self._safe_convert_to_float(system_stats.get('max_cpu')),
                self._safe_convert_to_float(system_stats.get('avg_memory')),
                self._safe_convert_to_float(system_stats.get('max_memory')),
                self._safe_convert_to_float(system_stats.get('avg_gpu')),
                self._safe_convert_to_float(system_stats.get('max_gpu')),
                str(status),
                str(error_message) if error_message else None,
                run_id
            )
            cursor.execute(query, params)
            return True
        except pg8000.dbapi.DatabaseError as e:
            self.logger.error(f"Fehler beim Aktualisieren des Runs: {e}")
            return False
        finally:
            cursor.close()
    
    def close(self):
        """Schließt die Datenbankverbindung"""
        if self.connection:
            self.connection.close()
            print("✓ Datenbankverbindung geschlossen")
