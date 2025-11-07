#!/usr/bin/env python3
"""
Verbesserte Zeitreihenanalyse mit vereinfachter Bewegungserkennung
MIT AUSFÜHRLICHEM LOGGING UND ROBUSTEM FEHLERHANDLING
"""

import time
from typing import Dict
from datetime import datetime

from DatabaseHandler import DatabaseHandler
from MovementDetector import MovementDetector
from RoomOccupancyManager import RoomOccupancyManager


class TimeSeriesAnalyzer:
    """
    Verbesserte Zeitreihenanalyse mit vereinfachter Bewegungserkennung
    """
    
    def __init__(self, db_config: Dict[str, str]):
        """
        Initialisiert den Analyzer
        """
        self.db = DatabaseHandler(**db_config)
        self.movement_detector = MovementDetector(
            transition_window=10.0,  # Erhöht für mehr Flexibilität
            min_confidence=0.3       # Gesenkt für mehr Detections
        )
        self.occupancy_manager = None  # Wird nach DB-Connect initialisiert
        
        # Analyse-Parameter
        self.analysis_window = 30.0
        self.min_detections = 1
    
    def start(self, interval_seconds: int = 30, continuous: bool = True):
        """
        Startet kontinuierliche Analyse
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
        print(f"📊 VERBESSERTE BEWEGUNGSANALYSE GESTARTET")
        print(f"{'='*80}")
        print(f"  Analysefenster: {self.analysis_window}s")
        print(f"  Update-Intervall: {interval_seconds}s")
        print(f"  Initiale Belegung: {self.occupancy_manager.get_current_occupancy()} Personen")
        print(f"  Min. Detections: {self.min_detections}")
        print(f"  Kontinuierlich: {'Ja' if continuous else 'Nein'}")
        print(f"{'='*80}\n")
        
        analysis_count = 0
        
        try:
            while True:
                analysis_count += 1
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                print(f"\n{'='*80}")
                print(f"[Analyse #{analysis_count}] {timestamp}")
                print(f"{'='*80}")
                
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
        Führt einen vollständigen Analysezyklus mit Logging aus
        """
        print(f"\n📥 Hole unverarbeitete Detections (Fenster: {self.analysis_window}s)...")
        
        recent = self.db.get_unprocessed_detections(self.analysis_window)
        print(f"   → Erhalten: {len(recent)} Detections")

        if len(recent) < self.min_detections:
            print(f"⏳ Zu wenig Detections ({len(recent)}/{self.min_detections})")
            return

        # Quellen zählen
        x_count = len([d for d in recent if d['source'] == 'input_x'])
        y_count = len([d for d in recent if d['source'] == 'input_y'])
        other_count = len(recent) - x_count - y_count
        
        print(f"\n📊 Detection-Verteilung:")
        print(f"   Input X: {x_count}")
        print(f"   Input Y: {y_count}")
        if other_count > 0:
            print(f"   Andere: {other_count}")

        # Beispiel-Detections (mit robustem Format)
        print(f"\n📸 Beispiel-Detections:")
        for d in recent[:5]:
            time_str = d.get('timestamp')
            if isinstance(time_str, datetime):
                time_str = time_str.strftime('%H:%M:%S')
            else:
                time_str = str(time_str) or "N/A"
            
            source = str(d.get('source', 'unknown'))
            persons = d.get('persons_detected')
            if not isinstance(persons, int):
                persons = 0
            avg_conf = d.get('avg_confidence')
            if avg_conf is None:
                avg_conf = 0.0

            print(f"   {time_str:>8s} | {source:10s} | {persons:2d} Pers. | Conf: {avg_conf:.3f}")

        if len(recent) > 5:
            print(f"   ... und {len(recent)-5} weitere")

        # Bewegungserkennung starten
        print(f"\n{'─'*80}")
        movements = self.movement_detector.detect_movements(recent)
        print(f"{'─'*80}")

        if not movements:
            print(f"\n✓ Keine gültigen Bewegungen erkannt")
            self.db.mark_detections_processed([d['id'] for d in recent])
            return

        print(f"\n🎯 {len(movements)} Bewegung(en) zur Verarbeitung:")

        for i, movement in enumerate(movements, 1):
            movement_type = movement.get('type', 'unknown')
            person_count = movement.get('person_count', 0)
            confidence = movement.get('confidence', 0.0)
            time_diff = movement.get('time_diff', 0)
            pattern = movement.get('pattern', 'N/A')
            x_delta = movement.get('x_delta', 0)
            y_delta = movement.get('y_delta', 0)

            print(f"\n  {'─'*76}")
            print(f"  [{i}] Typ: {movement_type.upper()}")
            print(f"      Personen: {person_count}")
            print(f"      Konfidenz: {confidence:.2f}")
            print(f"      Zeit-Diff: {time_diff:.2f}s")
            print(f"      Muster: {pattern}")
            print(f"      X-Delta: {x_delta}, Y-Delta: {y_delta}")

            movement_id = self.db.insert_movement(
                movement_type=movement_type,
                person_count=person_count,
                confidence=confidence,
                detection_sequence=movement.get('sequence', []),
                notes=f"Pattern: {pattern}"
            )

            if movement_id:
                print(f"      ✓ In DB gespeichert (ID: {movement_id})")
            else:
                print(f"      ✗ DB-Speicherung fehlgeschlagen")

            updated = self.occupancy_manager.process_movement(movement, movement_id)

            if not updated:
                print(f"      ⚠️  Nicht auf Raumzustand angewendet (Konfidenz/Plausibilität)")

        # Markiere als verarbeitet
        detection_ids = [d['id'] for d in recent]
        marked = self.db.mark_detections_processed(detection_ids)
        
        if marked:
            print(f"\n✓ {len(detection_ids)} Detections als verarbeitet markiert")
        else:
            print(f"\n✗ Fehler beim Markieren der Detections")

        current = self.occupancy_manager.get_current_occupancy()
        print(f"\n{'='*80}")
        print(f"📊 AKTUELLER RAUMZUSTAND: {current} Personen")
        print(f"{'='*80}")

    
    def _print_final_summary(self):
        """
        Gibt finale Zusammenfassung aus
        """
        print(f"\n{'='*80}")
        print(f"📋 ANALYSE BEENDET")
        print(f"{'='*80}")
        
        current = self.occupancy_manager.get_current_occupancy()
        print(f"  Finaler Raumzustand: {current} Personen")
        
        latest_state = self.db.get_latest_room_state()
        if latest_state:
            ts = latest_state.get('timestamp')
            reason = latest_state.get('change_reason', 'N/A')
            print(f"  Letztes Update: {ts}")
            print(f"  Grund: {reason}")
        
        print(f"{'='*80}\n")
    
    def get_current_occupancy(self) -> int:
        """
        Gibt aktuelle Personenanzahl im Raum zurück
        """
        if self.occupancy_manager:
            return self.occupancy_manager.get_current_occupancy()
        return 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Verbesserte Zeitreihenanalyse mit ausführlichem Logging'
    )
    
    parser.add_argument('--db-host', default='localhost', help='PostgreSQL Host')
    parser.add_argument('--db-user', required=True, help='PostgreSQL Benutzername')
    parser.add_argument('--db-password', required=True, help='PostgreSQL Passwort')
    parser.add_argument('--db-name', required=True, help='PostgreSQL Datenbankname')
    parser.add_argument('--db-port', type=int, default=5432, help='PostgreSQL Port')
    
    parser.add_argument('--interval', type=int, default=30, help='Analyse-Intervall (Sekunden)')
    parser.add_argument('--once', action='store_true', help='Nur eine Analyse durchführen')
    
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
