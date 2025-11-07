#!/usr/bin/env python3
"""
Verbesserte Zeitreihenanalyse mit Bewegungserkennung
Erkennt Ein- und Austritte und hält Raumzustand aktuell
"""

import time
from typing import Dict
from datetime import datetime

from DatabaseHandler import DatabaseHandler
from MovementDetector import MovementDetector
from RoomOccupancyManager import RoomOccupancyManager


class TimeSeriesAnalyzer:
    """
    Verbesserte Zeitreihenanalyse mit Bewegungserkennung
    """
    
    def __init__(self, db_config: Dict[str, str]):
        """
        Initialisiert den Analyzer
        
        Args:
            db_config: Datenbank-Konfiguration
        """
        self.db = DatabaseHandler(**db_config)
        self.movement_detector = MovementDetector(
            transition_window=3.0,
            min_confidence=0.4
        )
        self.occupancy_manager = None  # Wird nach DB-Connect initialisiert
        
        # Analyse-Parameter - ANGEPASST für weniger "Zu wenig Detections" Fehler
        self.analysis_window = 30.0  # Sekunden für Analysefenster (erhöht von 5s)
        self.min_detections = 1  # Mindestanzahl Detections pro Fenster (reduziert von 2)
    
    def start(self, interval_seconds: int = 30, continuous: bool = True):
        """
        Startet kontinuierliche Analyse
        
        Args:
            interval_seconds: Intervall zwischen Analysen (Standard 30s)
            continuous: True für kontinuierlichen Betrieb
        """
        if not self.db.connect():
            print("✗ Datenbankverbindung fehlgeschlagen")
            return
        
        if not self.db.create_tables():
            print("✗ Tabellenerstellung fehlgeschlagen")
            return
        
        # Occupancy Manager initialisieren
        self.occupancy_manager = RoomOccupancyManager(self.db, max_capacity=100)
        self.occupancy_manager.initialize()
        
        print(f"\n{'='*80}")
        print(f"📊 BEWEGUNGSANALYSE GESTARTET")
        print(f"{'='*80}")
        print(f"  Analysefenster: {self.analysis_window}s")
        print(f"  Update-Intervall: {interval_seconds}s")
        print(f"  Initiale Belegung: {self.occupancy_manager.get_current_occupancy()} Personen")
        print(f"  Kontinuierlich: {'Ja' if continuous else 'Nein'}")
        print(f"{'='*80}\n")
        
        analysis_count = 0
        
        try:
            while True:
                analysis_count += 1
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                print(f"\n[Analyse #{analysis_count}] {timestamp}")
                print("-" * 80)
                
                self._analyze_cycle()
                
                if not continuous:
                    break
                
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\n\n❌ Analyse durch Benutzer abgebrochen")
        finally:
            self._print_final_summary()
            self.db.close()
    
    def _analyze_cycle(self):
        """
        Ein Analyse-Zyklus
        """
        # Hole unverarbeitete Detections aus Zeitfenster
        recent = self.db.get_unprocessed_detections(self.analysis_window)

        if len(recent) < self.min_detections:
            print(f"⏳ Zu wenig Detections ({len(recent)}/{self.min_detections})")
            return

        print(f"🔍 Analysiere {len(recent)} Detections...")

        # Separiere nach Quelle für Info
        x_count = len([d for d in recent if d['source'] == 'input_x'])
        y_count = len([d for d in recent if d['source'] == 'input_y'])
        print(f"   └─ Input X: {x_count}, Input Y: {y_count}")

        # Erkenne Bewegungen
        movements = self.movement_detector.detect_movements(recent)

        if not movements:
            print(f"✓ Keine gültigen Bewegungen erkannt (fehlerhafte Detections übersprungen)")
            self.db.mark_detections_processed([d['id'] for d in recent])
            return

        # Verarbeite erkannte Bewegungen
        print(f"\n🎯 {len(movements)} Bewegung(en) erkannt:")

        for i, movement in enumerate(movements, 1):
            movement_type = movement['type']
            person_count = movement['person_count']
            confidence = movement['confidence']
            time_diff = movement.get('time_diff', 0)

            print(f"\n  [{i}] Typ: {movement_type.upper()}")
            print(f"      Personen: {person_count}")
            print(f"      Konfidenz: {confidence:.2f}")
            print(f"      Zeit-Diff: {time_diff:.2f}s")
            print(f"      X-Delta: {movement['x_delta']}, Y-Delta: {movement['y_delta']}")

            movement_id = self.db.insert_movement(
                movement_type=movement_type,
                person_count=person_count,
                confidence=confidence,
                detection_sequence=movement['sequence'],
                notes=f"Time diff: {time_diff:.2f}s"
            )

            updated = self.occupancy_manager.process_movement(movement, movement_id)

            if not updated:
                print(f"      Status: ⚠️  Nicht angewendet (Konfidenz oder Plausibilität)")

        # Markiere Detections als verarbeitet
        detection_ids = [d['id'] for d in recent]
        self.db.mark_detections_processed(detection_ids)

        current = self.occupancy_manager.get_current_occupancy()
        print(f"\n📊 Aktuell im Raum: {current} Personen")

    
    def _print_final_summary(self):
        """
        Gibt finale Zusammenfassung aus
        """
        print(f"\n{'='*80}")
        print(f"📋 ANALYSE BEENDET")
        print(f"{'='*80}")
        
        current = self.occupancy_manager.get_current_occupancy()
        print(f"  Finaler Raumzustand: {current} Personen")
        
        # Hole Statistiken
        latest_state = self.db.get_latest_room_state()
        if latest_state:
            print(f"  Letztes Update: {latest_state['timestamp']}")
            print(f"  Grund: {latest_state['change_reason']}")
        
        print(f"{'='*80}\n")
    
    def get_current_occupancy(self) -> int:
        """
        Gibt aktuelle Personenanzahl im Raum zurück
        
        Returns:
            Anzahl Personen
        """
        if self.occupancy_manager:
            return self.occupancy_manager.get_current_occupancy()
        return 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Verbesserte Zeitreihenanalyse für Personenzählungen'
    )
    
    # Datenbank-Konfiguration
    parser.add_argument('--db-host', default='localhost', help='PostgreSQL Host')
    parser.add_argument('--db-user', required=True, help='PostgreSQL Benutzername')
    parser.add_argument('--db-password', required=True, help='PostgreSQL Passwort')
    parser.add_argument('--db-name', required=True, help='PostgreSQL Datenbankname')
    parser.add_argument('--db-port', type=int, default=5432, help='PostgreSQL Port')
    
    # Analyse-Konfiguration
    parser.add_argument('--interval', type=int, default=30, 
                       help='Analyse-Intervall (Sekunden)')
    parser.add_argument('--once', action='store_true', 
                       help='Nur eine Analyse durchführen')
    
    args = parser.parse_args()
    
    db_config = {
        'host': args.db_host,
        'user': args.db_user,
        'password': args.db_password,
        'database': args.db_name,
        'port': args.db_port
    }
    
    analyzer = TimeSeriesAnalyzer(db_config)
    analyzer.start(
        interval_seconds=args.interval,
        continuous=not args.once
    )