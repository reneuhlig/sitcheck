#!/usr/bin/env python3
"""
Verbessertes Hauptscript für die Personenerkennung mit verschiedenen KI-Modellen
Jetzt mit CSV-Export und verbesserter Fehlerbehandlung
"""

import argparse
import sys
import os
from typing import Optional, List
import logging

# Import der Detector-Implementierungen
from UltralyticsPersonDetector import UltralyticsPersonDetector
from DetectionProcessor import DetectionProcessor


# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_detector(model_name: str, **kwargs):

    """
    Verbesserte Factory function für Detector-Erstellung mit Fehlerbehandlung
    
    Args:
        model_name: Name des Modells ('ultralytics', 'deepface', 'gemma', 'ollama-gemma3')
        **kwargs: Zusätzliche Argumente für den Detector
    """

    try:
        if model_name.lower() == 'ultralytics':
            return UltralyticsPersonDetector(
                model_path=kwargs.get('model_path', 'yolov8n.pt'),
                confidence_threshold=kwargs.get('confidence_threshold', 0.5)
            )
        
      
    except Exception as e:
        logger.error(f"Fehler bei der Detector-Erstellung für {model_name}: {e}")
        raise


def validate_arguments(args) -> bool:
    """Validiert die Kommandozeilenargumente"""
    # Datenverzeichnis prüfen
    if not os.path.exists(args.data_dir):
        logger.error(f"Datenverzeichnis nicht gefunden: {args.data_dir}")
        return False
    
    if not os.path.isdir(args.data_dir):
        logger.error(f"Datenverzeichnis ist keine Datei: {args.data_dir}")
        return False
    
    # Confidence-Threshold prüfen
    if not 0.0 <= args.confidence_threshold <= 1.0:
        logger.error(f"Confidence-Threshold muss zwischen 0.0 und 1.0 liegen: {args.confidence_threshold}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Personenerkennung mit verschiedenen KI-Modellen (mit CSV-Export)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Datenbank-Konfiguration
    parser.add_argument('--db-host', default='localhost', help='MySQL Host')
    parser.add_argument('--db-user', required=True, help='MySQL Benutzername')
    parser.add_argument('--db-password', required=True, help='MySQL Passwort')
    parser.add_argument('--db-name', required=True, help='MySQL Datenbankname')
    
    # Daten-Konfiguration
    parser.add_argument('--data-dir', required=True, help='Pfad zu klassifizierten Bilddaten')
    parser.add_argument('--max-images', type=int, help='Maximale Anzahl zu verarbeitender Bilder')
    parser.add_argument('--classifications', nargs='+', help='Nur bestimmte Klassifizierungen verarbeiten')
    parser.add_argument('--no-randomize', action='store_true', help='Deaktiviert Randomisierung der Bilderreihenfolge')
    
    # Modell-spezifische Optionen
    parser.add_argument('--confidence-threshold', type=float, default=0.5, help='Mindest-Konfidenz für Detections')
    
    # Ultralytics-spezifisch
    parser.add_argument('--yolo-model-path', default='yolov8n.pt', help='Pfad zum YOLO Modell')
    
    # Run-Konfiguration
    parser.add_argument('--run-name', help='Name für diesen Run (für Tracking)')
    parser.add_argument('--job-id', help='Job-ID für Cronjob-Tracking')
    
    # Debug-Optionen
    parser.add_argument('--verbose', '-v', action='store_true', help='Detaillierte Ausgabe')
    parser.add_argument('--dry-run', action='store_true', help='Testlauf ohne echte Verarbeitung')
    
    args = parser.parse_args()
    
    # Logging-Level anpassen
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Argumente validieren
    if not validate_arguments(args):
        sys.exit(1)
    
    # Dry-run Modus
    if args.dry_run:
        print("🧪 DRY-RUN Modus aktiviert - keine echte Verarbeitung")
        print(f"  Modell: {args.model}")
        print(f"  Datenverzeichnis: {args.data_dir}")
        print(f"  Max. Bilder: {args.max_images or 'Alle'}")
        return
    
    # Datenbankverbindung konfigurieren
    db_config = {
        'host': args.db_host,
        'user': args.db_user,
        'password': args.db_password,
        'database': args.db_name
    }
    
    # Run-Konfiguration erstellen
    run_config = {
        'run_name': args.run_name,
        'job_id': args.job_id,
        'script_args': vars(args),
        'confidence_threshold': args.confidence_threshold
    }
    
    try:
        # Detector erstellen
        detector_kwargs = {
            'confidence_threshold': args.confidence_threshold
        }
        
        if args.model == 'ultralytics':
            detector_kwargs['model_path'] = args.yolo_model_path
        
        print(f"🚀 Initialisiere {args.model} Detektor...")
        detector = create_detector(args.model, **detector_kwargs)
        
        processor = DetectionProcessor(
            detector=detector,
            db_config=db_config,
            data_dir=args.data_dir,
            run_config=run_config,
        )
        
        # Verarbeitung starten
        print("\n" + "="*80)
        print("🔍 PERSONENERKENNUNG GESTARTET")
        print("="*80)
        
        run_id = processor.process_images(
            max_images=args.max_images,
            classifications=args.classifications,
            randomize=not args.no_randomize
        )
        
        print(f"\n✅ Run erfolgreich abgeschlossen: {run_id}")
        
    except KeyboardInterrupt:
        print("\n❌ Verarbeitung durch Benutzer abgebrochen")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Kritischer Fehler: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()