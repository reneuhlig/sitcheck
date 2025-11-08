#!/usr/bin/env python3
"""
Time Series Analyzer - Bewegungsanalyse aus zeitbasierten Detections
Erkennt Entry/Exit-Muster und aktualisiert Raumzustand
"""

import time
from typing import Dict
from datetime import datetime

from DatabaseHandler import DatabaseHandler
from MovementDetector import MovementDetector
from RoomOccupancyManager import RoomOccupancyManager


class TimeSeriesAnalyzer:
    """Zeitreihenanalyse mit Bewegungserkennung"""
    
    def __init__(self, db_config: Dict[str, str]):
        """
        Initialisiert den Analyzer
        
        Args:
            db_config: Datenbank-Konfiguration
        """
        self.db = DatabaseHandler(**db_config)
        self.movement_detector = MovementDetector(
            transition_window=10.0,
            min_confidence=0.3
        )
        self.occupancy_manager = None  # Wird nach DB-Connect initialisiert
        
        # Analyse-Parameter
        self.analysis_window = 30.0
        self.min_detections = 1
    
    def start(self, interval_seconds: int = 30, continuous: bool = True):
        """
        Startet kontinuierliche Analyse
        
        Args:
            interval_seconds: Intervall zwischen Analysen
            continuous: True fuer kontinuierliche Ausfuehrung
        """
        if not self.db.connect():
            print("[ERROR] Datenbankverbindung fehlgeschlagen")
            return
        
        if not self.db.create_tables():
            print("[ERROR] Tabellenerstellung fehlgeschlagen")
            return
        
        # Occupancy Manager initialisieren
        self.occupancy_manager = RoomOccupancyManager(self.db, max_capacity=100)
        self.occupancy_manager.initialize()
        
        print(f"\n{'='*80}")
        print(f"[ANALYZER] BEWEGUNGSANALYSE GESTARTET")
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
                
                print(f"\n{'='*80}")
                print(f"[ANALYZER] Zyklus #{analysis_count} - {timestamp}")
                print(f"{'='*80}")
                
                self._analyze_cycle()
                
                if not continuous:
                    break
                
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\n\n[INFO] Analyse durch Benutzer abgebrochen")
        finally:
            self._print_final_summary()
            self.db.close()
    
    def _analyze_cycle(self):
        """Fuehrt einen vollstaendigen Analysezyklus durch"""
        # Hole unverarbeitete Detections
        recent = self.db.get_unprocessed_detections(self.analysis_window)
        print(f"[ANALYZER] Unverarbeitete Detections: {len(recent)}")

        if len(recent) < self.min_detections:
            print(f"[ANALYZER] Zu wenig Detections ({len(recent)}/{self.min_detections})")
            return

        # Quellen zaehlen
        x_count = len([d for d in recent if d['source'] == 'input_x'])
        y_count = len([d for d in recent if d['source'] == 'input_y'])
        print(f"[ANALYZER] Input X: {x_count} | Input Y: {y_count}")

        # Zeige Beispiel-Detections (nur erste 3)
        if len(recent) > 0:
            print(f"[ANALYZER] Erste Detections:")
            for d in recent[:3]:
                time_str = d.get('timestamp')
                if isinstance(time_str, datetime):
                    time_str = time_str.strftime('%H:%M:%S')
                else:
                    time_str = str(time_str) or "N/A"
                
                source = str(d.get('source', 'unknown'))
                persons = d.get('persons_detected', 0)
                avg_conf = d.get('avg_confidence') or 0.0

                print(f"  {time_str:>8s} | {source:10s} | {persons:2d} Pers | Conf={avg_conf:.3f}")
            
            if len(recent) > 3:
                print(f"  ... und {len(recent)-3} weitere")

        # Bewegungserkennung starten
        print(f"\n[ANALYZER] Starte Bewegungsanalyse...")
        movements = self.movement_detector.detect_movements(recent)

        if not movements:
            print(f"[ANALYZER] Keine Bewegungen erkannt")
            self.db.mark_detections_processed([d['id'] for d in recent])
            return

        print(f"\n[ANALYZER] {len(movements)} Bewegung(en) verarbeiten:")

        # Verarbeite jede erkannte Bewegung
        for i, movement in enumerate(movements, 1):
            movement_type = movement.get('type', 'unknown')
            person_count = movement.get('person_count', 0)
            confidence = movement.get('confidence', 0.0)
            pattern = movement.get('pattern', 'N/A')

            print(f"\n  [{i}] {movement_type.upper()}")
            print(f"      Personen: {person_count}")
            print(f"      Konfidenz: {confidence:.2f}")
            print(f"      Muster: {pattern}")

            # Speichere Bewegung in DB
            movement_id = self.db.insert_movement(
                movement_type=movement_type,
                person_count=person_count,
                confidence=confidence,
                detection_sequence=movement.get('sequence', []),
                notes=f"Pattern: {pattern}"
            )

            if movement_id:
                print(f"      Gespeichert: ID={movement_id}")
            else:
                print(f"      [ERROR] Speicherung fehlgeschlagen")

            # Aktualisiere Raumzustand
            updated = self.occupancy_manager.process_movement(movement, movement_id)

            if not updated:
                print(f"      Raumzustand nicht aktualisiert (Konfidenz/Plausibilitaet)")

        # Markiere als verarbeitet
        detection_ids = [d['id'] for d in recent]
        marked = self.db.mark_detections_processed(detection_ids)
        
        if marked:
            print(f"\n[ANALYZER] {len(detection_ids)} Detections als verarbeitet markiert")
        else:
            print(f"\n[ANALYZER] [ERROR] Fehler beim Markieren")

        # Zeige aktuellen Raumzustand
        current = self.occupancy_manager.get_current_occupancy()
        print(f"\n{'='*80}")
        print(f"[ROOM] AKTUELLER ZUSTAND: {current} Personen im Raum")
        print(f"{'='*80}")

    
    def _print_final_summary(self):
        """Gibt finale Zusammenfassung aus"""
        print(f"\n{'='*80}")
        print(f"[ANALYZER] ANALYSE BEENDET")
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
        Gibt aktuelle Personenanzahl im Raum zurueck
        
        Returns:
            Anzahl Personen
        """
        if self.occupancy_manager:
            return self.occupancy_manager.get_current_occupancy()
        return 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Zeitreihenanalyse mit Bewegungserkennung'
    )
    
    parser.add_argument('--db-host', default='localhost', help='PostgreSQL Host')
    parser.add_argument('--db-user', required=True, help='PostgreSQL Benutzername')
    parser.add_argument('--db-password', required=True, help='PostgreSQL Passwort')
    parser.add_argument('--db-name', required=True, help='PostgreSQL Datenbankname')
    parser.add_argument('--db-port', type=int, default=5432, help='PostgreSQL Port')
    
    parser.add_argument('--interval', type=int, default=30, help='Analyse-Intervall (Sekunden)')
    parser.add_argument('--once', action='store_true', help='Nur eine Analyse durchfuehren')
    
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