import time
import uuid
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import numpy as np

from BaseDetector import BaseDetector
from DataLoader import LiveImageLoader
from DatabaseHandler import DatabaseHandler


class DetectionProcessor:
    """Verbesserte Hauptklasse für die Verarbeitung mit beliebigen Detektoren"""
    
    def __init__(self, detector: BaseDetector, db_config: Dict[str, str], 
                 data_dir: str, run_config: Dict[str, Any] = None):
        """
        Initialisiert den Detection Processor
        
        Args:
            detector: Instanz eines BaseDetector
            db_config: MySQL Konfiguration
            data_dir: Pfad zu klassifizierten Daten
            run_config: Zusätzliche Konfiguration für den Run
        """
        self.detector = detector
        self.db = DatabaseHandler(**db_config)
        self.data_loader = LiveImageLoader(data_dir)
        self.run_config = run_config or {}
    
        
    def process_images(self, max_images: Optional[int] = None,
                      classifications: List[str] = None,
                      randomize: bool = True) -> str:
        """
        Verarbeitet Bilder mit dem konfigurierten Detektor
        
        Args:
            max_images: Maximale Anzahl zu verarbeitender Bilder
            classifications: Nur bestimmte Klassifizierungen verarbeiten
            randomize: Reihenfolge randomisieren
            
        Returns:
            Run-ID für diesen Durchlauf
        """
        # Run-ID generieren
        run_id = str(uuid.uuid4())
        # Datenbankverbindung und Setup
        db_connected = self._setup_database(run_id)
        
        print(f"✓ Neuer Run gestartet: {run_id}")
        print(f"  Modell: {self.detector.model_name} v{self.detector.model_version}")
        print(f"  Datenbank: {'✓' if db_connected else '✗'}")
        
        # Bilder laden
        images = self.data_loader.get_classified_images(
            randomize=randomize, 
            classifications=classifications
        )
        
        if not images:
            print("✗ Keine Bilder zum Verarbeiten gefunden")
            return run_id
            
        if max_images:
            images = images[:max_images]
            print(f"  Limitiert auf {max_images} Bilder")
        
        # Verarbeitung durchführen
        return self._execute_processing(run_id, images, db_connected)
    
    def _setup_database(self, run_id: str) -> bool:
        """Setzt Datenbank auf und erstellt Run-Eintrag"""
        try:
            if not self.db.connect():
                return False
                
            if not self.db.create_tables():
                return False
                
            # Run in Datenbank erstellen
            model_info = self.detector.get_model_info()
            full_config = {
                **self.run_config,
                'model_info': model_info
            }
            
            return self.db.insert_run(run_id, self.detector.model_name, 
                                     self.detector.model_version, full_config)
        except Exception as e:
            print(f"⚠ Datenbank-Setup fehlgeschlagen: {e}")
            return False
    
    def _execute_processing(self, run_id: str, images: List[Tuple[str, str]], 
                          db_connected: bool) -> str:
        """Führt die eigentliche Bildverarbeitung durch"""
        # Monitoring starten
        self.monitor.start_monitoring()
        
        # Statistiken
        processing_times = []
        successful_detections = 0
        failed_detections = 0
        start_time = time.time()
        
        print(f"\nStarte Verarbeitung von {len(images)} Bildern...")
        print("-" * 80)
        
        status = 'completed'
        
        try:
            for i, (image_path, classification) in enumerate(images, 1):
                result_data = self._process_single_image(
                    run_id, i, len(images), image_path, classification
                )
                
                processing_times.append(result_data['processing_time'])
                
                if result_data['success']:
                    successful_detections += 1
                else:
                    failed_detections += 1
                
                # In Datenbank und CSV speichern
                self._save_result(result_data, db_connected)
                
                # Kurze Pause zwischen den Bildern
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\n❌ Verarbeitung durch Benutzer abgebrochen")
            status = 'cancelled'
        except Exception as e:
            print(f"\n❌ Kritischer Fehler: {e}")
            status = 'failed'
        
        # Verarbeitung abschließen
        return self._finalize_processing(
            run_id, images, processing_times, successful_detections, 
            failed_detections, start_time, status, db_connected
        )
    
    def _process_single_image(self, run_id: str, current: int, total: int, 
                            image_path: str, classification: str) -> Dict[str, Any]:
        """Verarbeitet ein einzelnes Bild"""
        detection_start = time.time()
        
        try:
            # Detection durchführen
            result = self.detector.detect(image_path)
            processing_time = time.time() - detection_start
            
            # Bildinformationen
            image_info = self.data_loader.get_image_info(image_path)
            
            # Status ausgeben
            self._print_detection_result(current, total, image_info['filename'],
                                       classification, result, processing_time)
            
            return {
                'run_id': run_id,
                'image_path': image_path,
                'image_filename': image_info['filename'],
                'classification': classification,
                'model_output': result,
                'confidence_scores': self._format_confidences(result.get('confidences', [])),
                'processing_time': processing_time,
                'success': True,
                'error_message': None
            }
            
        except Exception as e:
            processing_time = time.time() - detection_start
            image_info = self.data_loader.get_image_info(image_path)
            
            print(f"[{current:4d}/{total}] {image_info['filename']} "
                  f"({classification}) -> FEHLER: {e}")
            
            return {
                'run_id': run_id,
                'image_path': image_path,
                'image_filename': image_info['filename'],
                'classification': classification,
                'model_output': None,
                'confidence_scores': "",
                'processing_time': processing_time,
                'success': False,
                'error_message': str(e)
            }
    
    def _save_result(self, result_data: Dict[str, Any], db_connected: bool):
        """Speichert Ergebnis in Datenbank und CSV"""
        # In Datenbank speichern
        if db_connected:
            try:
                self.db.insert_result(**result_data)
            except Exception as e:
                print(f"⚠ DB-Speicherung fehlgeschlagen: {e}")
        
        # In CSV speichern
        if self.results_csv_path:
            try:
                self.csv_exporter.write_result(self.results_csv_path, result_data)
            except Exception as e:
                print(f"⚠ CSV-Speicherung fehlgeschlagen: {e}")
    
    def _finalize_processing(self, run_id: str, images: List, processing_times: List[float],
                           successful: int, failed: int, start_time: float, 
                           status: str, db_connected: bool) -> str:
        """Schließt die Verarbeitung ab und speichert Zusammenfassung"""
        # Monitoring stoppen
        self.monitor.stop_monitoring()
        
        # Statistiken berechnen
        total_time = time.time() - start_time
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        system_stats = self.monitor.get_average_usage()
        
        # Run-Informationen für DB und CSV vorbereiten
        run_completion_data = {
            'run_id': run_id,
            'model_name': self.detector.model_name,
            'model_version': self.detector.model_version,
            'start_time': time.time() - total_time,  # Rückberechnung
            'end_time': time.time(),
            'total_images': len(images),
            'successful_detections': successful,
            'failed_detections': failed,
            'avg_processing_time': avg_processing_time,
            'total_processing_time': total_time,
            'system_stats': system_stats,
            'status': status,
            'error_message': None,
            'config': self.run_config
        }
        
        # In Datenbank aktualisieren
        if db_connected:
            try:
                self.db.update_run_completion(
                    run_id=run_id,
                    total_images=len(images),
                    successful_detections=successful,
                    failed_detections=failed,
                    avg_processing_time=avg_processing_time,
                    total_processing_time=total_time,
                    system_stats=system_stats,
                    status=status
                )
            except Exception as e:
                print(f"⚠ DB-Update fehlgeschlagen: {e}")
        
        # CSV-Run-Info speichern
        if self.run_info_csv_path:
            try:
                self.csv_exporter.write_run_info(self.run_info_csv_path, run_completion_data)
            except Exception as e:
                print(f"⚠ CSV-Run-Info speichern fehlgeschlagen: {e}")
        
        # Zusammenfassung ausgeben
        self._print_summary(run_id, len(images), successful, failed, 
                           total_time, avg_processing_time, system_stats)
        
        # Datenbankverbindung schließen
        if db_connected:
            self.db.close()
        
        return run_id
    
    def _format_confidences(self, confidences: List[float]) -> str:
        """Formatiert Konfidenzwerte als String"""
        if not confidences:
            return ""
        return ",".join([f"{c:.3f}" for c in confidences])
    
    def _print_detection_result(self, current: int, total: int, filename: str,
                               classification: str, result: Dict[str, Any],
                               processing_time: float):
        """Gibt Erkennungsergebnis auf Konsole aus"""
        persons = result.get('persons_detected', 0)
        conf = result.get('avg_confidence', 0.0)
        uncertain = "⚠" if result.get('uncertain', False) else "✓"
        
        print(f"[{current:4d}/{total}] {filename} ({classification}) -> "
              f"{persons} Personen, Konfidenz: {conf:.3f} {uncertain}, Zeit: {processing_time:.3f}s")
    
    def _print_summary(self, run_id: str, total_images: int, successful: int,
                      failed: int, total_time: float, avg_time: float,
                      system_stats: Dict[str, float]):
        """Gibt Zusammenfassung auf Konsole aus"""
        print("-" * 80)
        print(f"✓ Verarbeitung abgeschlossen!")
        print(f"  Run-ID: {run_id}")
        print(f"  Modell: {self.detector.model_name} v{self.detector.model_version}")
        print(f"  Verarbeitete Bilder: {successful + failed}/{total_images}")
        print(f"  Erfolgreiche Detections: {successful}")
        print(f"  Fehlgeschlagene Detections: {failed}")
        print(f"  Gesamtzeit: {total_time:.1f}s")
        print(f"  Durchschnittliche Zeit pro Bild: {avg_time:.3f}s")
        print(f"  Durchschnittliche CPU-Auslastung: {system_stats['avg_cpu']:.1f}%")
        print(f"  Maximale CPU-Auslastung: {system_stats['max_cpu']:.1f}%")
        print(f"  Durchschnittliche RAM-Auslastung: {system_stats['avg_memory']:.1f}%")
        if system_stats['avg_gpu'] > 0:
            print(f"  Durchschnittliche GPU-Auslastung: {system_stats['avg_gpu']:.1f}%")
        
        # CSV-Export-Info
        if self.results_csv_path:
            print(f"  CSV-Ergebnisse: {self.results_csv_path}")
        if self.run_info_csv_path:
            print(f"  CSV-Run-Info: {self.run_info_csv_path}")