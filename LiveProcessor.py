import time
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from DataLoader import LiveImageLoader
from DatabaseHandler import DatabaseHandler
from BaseDetector import BaseDetector


class LiveProcessor:
    """Live-Verarbeitung von Bildern aus zwei Ordnern"""
    
    def __init__(self,
                 detector: BaseDetector,
                 db_config: Dict[str, str],
                 input_x: str,
                 input_y: str,
                 poll_interval: float = 0.5):
        """
        Initialisiert den Live-Processor
        
        Args:
            detector: Detector-Instanz (z.B. UltralyticsPersonDetector)
            db_config: Datenbank-Konfiguration
            input_x: Pfad zu Ordner X
            input_y: Pfad zu Ordner Y
            poll_interval: Intervall zwischen Checks (Sekunden)
        """
        self.detector = detector
        self.db = DatabaseHandler(**db_config)
        self.input_x = Path(input_x)
        self.input_y = Path(input_y)
        self.poll_interval = poll_interval
        
        # Loader initialisieren
        self.loader = LiveImageLoader(
            str(self.input_x),
            str(self.input_y),
            poll_interval=poll_interval
        )
        
        self._running = False
    
    def start(self):
        """Startet die Live-Überwachung und Verarbeitung"""
        # Datenbank vorbereiten
        if not self.db.connect():
            print("✗ Datenbankverbindung fehlgeschlagen")
            return
        
        if not self.db.create_tables():
            print("✗ Tabellenerstellung fehlgeschlagen")
            return
        
        print(f"\n{'='*80}")
        print(f"🚀 LIVE-PERSONENERKENNUNG GESTARTET")
        print(f"{'='*80}")
        print(f"  Modell: {self.detector.model_name} v{self.detector.model_version}")
        print(f"  Ordner X: {self.input_x}")
        print(f"  Ordner Y: {self.input_y}")
        print(f"  Poll-Intervall: {self.poll_interval}s")
        print(f"{'='*80}\n")
        print("👀 Warte auf neue Bilder...\n")
        
        self._running = True
        processed_count = 0
        
        try:
            for source, img in self.loader.watch():
                if not self._running:
                    break
                
                if img is None:
                    continue
                
                processed_count += 1
                self._process_image(source, img, processed_count)
                
        except KeyboardInterrupt:
            print("\n\n❌ Verarbeitung durch Benutzer abgebrochen")
        finally:
            self.stop()
    
    def _process_image(self, source: str, img, count: int):
        """Verarbeitet ein einzelnes Bild"""
        start_time = time.time()
        timestamp = datetime.now()
        
        try:
            # Detection durchführen
            result = self.detector.detect(img)
            processing_time = time.time() - start_time
            
            # In Datenbank speichern
            detection_id = self.db.insert_detection(
                source=source,
                persons_detected=result['persons_detected'],
                avg_confidence=result['avg_confidence'],
                max_confidence=result['max_confidence'],
                min_confidence=result['min_confidence'],
                detection_data=result
            )
            
            # Status ausgeben
            status = "✓" if detection_id else "✗"
            print(f"[{count:4d}] {timestamp.strftime('%H:%M:%S.%f')[:-3]} | "
                  f"{source:10s} | {result['persons_detected']:2d} Personen | "
                  f"Konfidenz: {result['avg_confidence']:.3f} | "
                  f"Zeit: {processing_time:.3f}s {status}")
            
        except Exception as e:
            print(f"[{count:4d}] {timestamp.strftime('%H:%M:%S.%f')[:-3]} | "
                  f"{source:10s} | FEHLER: {e}")
    
    def stop(self):
        """Stoppt die Verarbeitung"""
        print("\n🛑 Stoppe Live-Verarbeitung...")
        self._running = False
        self.loader.stop()
        self.db.close()
        print("✓ Verarbeitung beendet")


if __name__ == "__main__":
    # Beispiel-Verwendung
    from UltralyticsPersonDetector import UltralyticsPersonDetector
    
    db_config = {
        'host': 'localhost',
        'user': 'aiuser',
        'password': 'DHBW1234!?',
        'database': 'ai_detection',
        'port': 5432
    }
    
    detector = UltralyticsPersonDetector(
        model_path="yolov8n.pt",
        confidence_threshold=0.5
    )
    
    processor = LiveProcessor(
        detector=detector,
        db_config=db_config,
        input_x="input_x",
        input_y="input_y",
        poll_interval=0.5
    )
    
    processor.start()