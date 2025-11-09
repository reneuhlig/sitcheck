#!/usr/bin/env python3
"""
Room Occupancy Manager - Verwaltung des Raumzustands (Personenanzahl)
"""

from typing import Dict, Optional
from DatabaseHandler import DatabaseHandler


class RoomOccupancyManager:
    """
    Verwaltet die Gesamtanzahl der Personen im Raum
    Implementiert Plausibilitaetspruefungen
    """
    
    def __init__(self, db: DatabaseHandler, max_capacity: int = 100):
        """
        Initialisiert den Occupancy Manager
        
        Args:
            db: DatabaseHandler Instanz
            max_capacity: Maximale Raumkapazitaet fuer Plausibilitaetspruefung
        """
        self.db = db
        self.max_capacity = max_capacity
        self.current_occupancy = 0
        self.initialized = False
    
    def initialize(self):
        """Laedt aktuellen Zustand aus Datenbank oder setzt auf 0"""
        latest_state = self.db.get_latest_room_state()
        
        if latest_state:
            self.current_occupancy = latest_state['total_persons']
            print(f"[ROOM] Zustand geladen: {self.current_occupancy} Personen")
        else:
            self.current_occupancy = 0
            self.db.insert_room_state(
                total_persons=0,
                change_reason='initialization',
                confidence=1.0,
                notes='Initialisierung des Systems'
            )
            print(f"[ROOM] Initialisiert: 0 Personen")
        
        self.initialized = True
    
    def process_movement(self, movement: Dict, movement_id: Optional[int] = None) -> bool:
        """
        Verarbeitet eine erkannte Bewegung und aktualisiert Raumzustand
        
        Args:
            movement: Movement-Dictionary vom MovementDetector
            movement_id: Optionale ID des gespeicherten Movement-Eintrags
            
        Returns:
            True wenn Zustand aktualisiert wurde, sonst False
        """

        
        if not self.initialized:
            print("[ROOM] Warnung: Manager nicht initialisiert")
            return False
        
        movement_type = movement['type']
        person_count = movement['person_count']
        confidence = movement['confidence']
        
        # Niedrige Konfidenz -> Ignorieren
        if confidence < 0.4:
            print(f"[ROOM] Bewegung ignoriert: Konfidenz zu niedrig ({confidence:.2f})")
            return False
        
        # Berechne neue Personenanzahl
        if movement_type == 'entry':
            new_count = self.current_occupancy + person_count
        elif movement_type == 'exit':
            new_count = max(0, self.current_occupancy - person_count)
        else:
            print(f"[ROOM] Warnung: Unbekannter Bewegungstyp: {movement_type}")
            return False
        
        # Plausibilitaetspruefung
        if not self._is_plausible(new_count, person_count):
            print(f"[ROOM] Implausible Aenderung: {self.current_occupancy} -> {new_count}")
            return False
        
        # Update durchfuehren
        old_count = self.current_occupancy
        self.current_occupancy = new_count
        
        # In Datenbank speichern
        success = self.db.insert_room_state(
            total_persons=new_count,
            change_reason=movement_type,
            movement_id=movement_id,
            confidence=confidence,
            notes=f"Delta: {person_count}, Zeit-Diff: {movement.get('time_diff', 0):.2f}s"
        )
        
        if success:
            # Komprimierte Log-Ausgabe
            sign = "+" if movement_type == 'entry' else "-"
            print(f"[ROOM] {movement_type.upper()}: {old_count} -> {new_count} "
                  f"({sign}{person_count}, Conf={confidence:.2f})")
            return True
        else:
            # Rollback bei Fehler
            self.current_occupancy = old_count
            print(f"[ROOM] Fehler beim Speichern des Raumzustands")
            return False
    
    def _is_plausible(self, new_count: int, change: int) -> bool:
        """
        Prueft Plausibilitaet einer Aenderung
        
        Args:
            new_count: Neue Gesamtanzahl
            change: Aenderung (Delta)
            
        Returns:
            True wenn plausibel, sonst False
        """
        # Nicht negativ
        if new_count < 0:
            return False
        
        # Nicht ueber Kapazitaet
        if new_count > self.max_capacity:
            print(f"[ROOM] Warnung: Ueber Maximalkapazitaet: {new_count} > {self.max_capacity}")
            return False
        
        # Nicht mehr als 10 Personen auf einmal (anpassbar)
        if abs(change) > 10:
            print(f"[ROOM] Warnung: Zu grosse Aenderung: {abs(change)} Personen")
            return False
        
        return True
    
    def get_current_occupancy(self) -> int:
        """
        Gibt aktuelle Personenanzahl im Raum zurueck
        
        Returns:
            Anzahl Personen
        """
        return self.current_occupancy
    
    def manual_correction(self, correct_count: int, reason: str = "manual_correction") -> bool:
        """
        Manuelle Korrektur des Zustands
        
        Args:
            correct_count: Korrekte Personenanzahl
            reason: Grund der Korrektur
            
        Returns:
            True bei Erfolg
        """
        if correct_count < 0 or correct_count > self.max_capacity:
            print(f"[ROOM] Warnung: Ungueltige Anzahl: {correct_count}")
            return False
        
        old_count = self.current_occupancy
        self.current_occupancy = correct_count
        
        success = self.db.insert_room_state(
            total_persons=correct_count,
            change_reason=reason,
            confidence=1.0,
            notes=f"Manuelle Korrektur von {old_count} auf {correct_count}"
        )
        
        if success:
            print(f"[ROOM] Manuelle Korrektur: {old_count} -> {correct_count}")
            return True
        else:
            self.current_occupancy = old_count
            return False
    
    def reset(self):
        """Setzt Raumzustand auf 0 zurueck (z.B. am Ende des Tages)"""
        return self.manual_correction(0, "manual_reset")