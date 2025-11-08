#!/usr/bin/env python3
"""
Hauptprogramm für Live-Personenerkennung
Überwacht zwei Ordner und verarbeitet Bilder in Echtzeit
"""

import argparse
import sys
import logging
from pathlib import Path

from UltralyticsPersonDetector import UltralyticsPersonDetector
from LiveProcessor import LiveProcessor


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_arguments(args) -> bool:
    """Validiert die Kommandozeilenargumente"""
    # Ordner prüfen/erstellen
    for folder in [args.input_x, args.input_y]:
        folder_path = Path(folder)
        if not folder_path.exists():
            try:
                folder_path.mkdir(parents=True, exist_ok=True)
                print(f"OK: Ordner erstellt: {folder}")
            except Exception as e:
                logger.error(f"Fehler beim Erstellen von {folder}: {e}")
                return False
    
    # Confidence-Threshold prüfen
    if not 0.0 <= args.confidence_threshold <= 1.0:
        logger.error(f"Confidence-Threshold muss zwischen 0.0 und 1.0 liegen: {args.confidence_threshold}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Live-Personenerkennung mit Ultralytics YOLO',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Datenbank-Konfiguration
    parser.add_argument('--db-host', default='localhost', help='PostgreSQL Host')
    parser.add_argument('--db-user', required=True, help='PostgreSQL Benutzername')
    parser.add_argument('--db-password', required=True, help='PostgreSQL Passwort')
    parser.add_argument('--db-name', required=True, help='PostgreSQL Datenbankname')
    parser.add_argument('--db-port', type=int, default=5432, help='PostgreSQL Port')
    
    # Ordner-Konfiguration
    parser.add_argument('--input-x', default='input_x', help='Pfad zu Eingabeordner X')
    parser.add_argument('--input-y', default='input_y', help='Pfad zu Eingabeordner Y')
    
    # Modell-Konfiguration
    parser.add_argument('--yolo-model', default='yolov8n.pt', help='Pfad zum YOLO Modell')
    parser.add_argument('--confidence-threshold', type=float, default=0.5, 
                       help='Mindest-Konfidenz für Detections')
    
    # Verarbeitungs-Konfiguration
    parser.add_argument('--poll-interval', type=float, default=0.5,
                       help='Intervall zwischen Ordner-Checks (Sekunden)')
    
    # Debug-Optionen
    parser.add_argument('--verbose', '-v', action='store_true', help='Detaillierte Ausgabe')
    
    args = parser.parse_args()
    
    # Logging-Level anpassen
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Argumente validieren
    if not validate_arguments(args):
        sys.exit(1)
    
    # Datenbankverbindung konfigurieren
    db_config = {
        'host': args.db_host,
        'user': args.db_user,
        'password': args.db_password,
        'database': args.db_name,
        'port': args.db_port
    }
    
    try:
        # Detector erstellen
        print(f" Initialisiere YOLO Detektor ({args.yolo_model})...")
        detector = UltralyticsPersonDetector(
            model_path=args.yolo_model,
            confidence_threshold=args.confidence_threshold
        )
        
        # Live-Processor erstellen
        processor = LiveProcessor(
            detector=detector,
            db_config=db_config,
            input_x=args.input_x,
            input_y=args.input_y,
            poll_interval=args.poll_interval
        )
        
        # Verarbeitung starten
        processor.start()
        
    except KeyboardInterrupt:
        print("\nERROR: Programm durch Benutzer abgebrochen")
        sys.exit(130)
    except Exception as e:
        logger.error(f"ERROR: Kritischer Fehler: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()