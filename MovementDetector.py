#!/usr/bin/env python3
"""
Bewegungserkennung für Personenzählung
Erkennt Ein- und Austritte basierend auf Kamera-Sequenzen
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class MovementDetector:
    """
    Erkennt Bewegungen (Ein-/Austritte) aus Detektions-Sequenzen
    """
    
    def __init__(self, 
                 transition_window: float = 3.0,
                 min_confidence: float = 0.4):
        """
        Initialisiert den Movement Detector
        
        Args:
            transition_window: Zeitfenster für Übergangserkennung (Sekunden)
            min_confidence: Mindest-Konfidenz für Detections
        """
        self.transition_window = transition_window
        self.min_confidence = min_confidence
    
    def detect_movements(self, detections: List[Dict]) -> List[Dict]:
        """
        Analysiert Detections und findet Bewegungsmuster
        
        Args:
            detections: Liste von Detection-Dictionaries
            
        Returns:
            Liste von erkannten Bewegungen
        """
        if len(detections) < 2:
            return []
        
        # Sortiere nach Zeit
        sorted_detections = sorted(detections, key=lambda x: x['timestamp'])
        
        # Separiere nach Quelle
        x_detections = [d for d in sorted_detections if d['source'] == 'input_x']
        y_detections = [d for d in sorted_detections if d['source'] == 'input_y']
        
        if not x_detections or not y_detections:
            return []
        
        # Analysiere Sequenzen
        movements = []
        
        # Analysiere X-Y Übergänge (Entry)
        entry_movement = self._detect_entry_pattern(x_detections, y_detections)
        if entry_movement:
            movements.append(entry_movement)
        
        # Analysiere Y-X Übergänge (Exit)
        exit_movement = self._detect_exit_pattern(x_detections, y_detections)
        if exit_movement:
            movements.append(exit_movement)
        
        return movements
    
    def _detect_entry_pattern(self, x_seq: List[Dict], y_seq: List[Dict]) -> Optional[Dict]:
        """
        Erkennt Eintritts-Muster:
        - X sieht Anstieg (Person erscheint außen)
        - Y sieht Anstieg (Person wird innen sichtbar)
        - X-Anstieg zeitlich vor Y-Anstieg
        
        Args:
            x_seq: Sequenz von X-Detections
            y_seq: Sequenz von Y-Detections
            
        Returns:
            Movement-Dictionary oder None
        """
        # Berechne Deltas
        x_delta = self._calculate_delta(x_seq)
        y_delta = self._calculate_delta(y_seq)
        
        # Entry-Bedingung: Beide zeigen Anstieg
        if x_delta <= 0 or y_delta <= 0:
            return None
        
        # Zeitliche Reihenfolge prüfen: X muss vor Y kommen
        x_first_increase = self._find_first_increase(x_seq)
        y_first_increase = self._find_first_increase(y_seq)
        
        if not x_first_increase or not y_first_increase:
            return None
        
        time_diff = (y_first_increase - x_first_increase).total_seconds()
        
        # X muss vor Y sein (aber innerhalb des Zeitfensters)
        if time_diff < 0 or time_diff > self.transition_window:
            return None
        
        # Bestimme Personenanzahl (nehme das Minimum für Konservativität)
        person_count = min(x_delta, y_delta)
        
        # Berechne Konfidenz
        confidence = self._calculate_confidence(x_seq, y_seq, time_diff)
        
        return {
            'type': 'entry',
            'person_count': person_count,
            'confidence': confidence,
            'time_diff': time_diff,
            'x_delta': x_delta,
            'y_delta': y_delta,
            'sequence': {
                'x_ids': [d['id'] for d in x_seq],
                'y_ids': [d['id'] for d in y_seq]
            }
        }
    
    def _detect_exit_pattern(self, x_seq: List[Dict], y_seq: List[Dict]) -> Optional[Dict]:
        """
        Erkennt Austritts-Muster:
        - Y sieht Abfall (Person verlässt Innenbereich)
        - X sieht Anstieg (Person erscheint außen)
        - Y-Abfall zeitlich vor X-Anstieg
        
        Args:
            x_seq: Sequenz von X-Detections
            y_seq: Sequenz von Y-Detections
            
        Returns:
            Movement-Dictionary oder None
        """
        # Berechne Deltas
        x_delta = self._calculate_delta(x_seq)
        y_delta = self._calculate_delta(y_seq)
        
        # Exit-Bedingung: Y zeigt Abfall, X zeigt Anstieg
        if x_delta <= 0 or y_delta >= 0:
            return None
        
        # Zeitliche Reihenfolge prüfen: Y-Abfall vor X-Anstieg
        x_first_increase = self._find_first_increase(x_seq)
        y_first_decrease = self._find_first_decrease(y_seq)
        
        if not x_first_increase or not y_first_decrease:
            return None
        
        time_diff = (x_first_increase - y_first_decrease).total_seconds()
        
        # Y muss vor X sein (aber innerhalb des Zeitfensters)
        if time_diff < 0 or time_diff > self.transition_window:
            return None
        
        # Bestimme Personenanzahl
        person_count = min(x_delta, abs(y_delta))
        
        # Berechne Konfidenz
        confidence = self._calculate_confidence(x_seq, y_seq, time_diff)
        
        return {
            'type': 'exit',
            'person_count': person_count,
            'confidence': confidence,
            'time_diff': time_diff,
            'x_delta': x_delta,
            'y_delta': y_delta,
            'sequence': {
                'x_ids': [d['id'] for d in x_seq],
                'y_ids': [d['id'] for d in y_seq]
            }
        }
    
    def _calculate_delta(self, sequence: List[Dict]) -> int:
        """
        Berechnet Änderung der Personenanzahl in einer Sequenz
        
        Vergleicht erste und letzte Detection mit ausreichender Konfidenz
        
        Args:
            sequence: Liste von Detection-Dictionaries
            
        Returns:
            Delta (positiv = Anstieg, negativ = Abfall)
        """
        if len(sequence) < 2:
            return 0
        
        # Filtere nach Konfidenz - FIX: Behandle None-Werte
        high_conf = [d for d in sequence 
                     if (d.get('avg_confidence') or 0.0) >= self.min_confidence]
        
        if len(high_conf) < 2:
            return 0
        
        start_count = high_conf[0]['persons_detected']
        end_count = high_conf[-1]['persons_detected']
        
        return end_count - start_count
    
    def _find_first_increase(self, sequence: List[Dict]) -> Optional[datetime]:
        """
        Findet Zeitpunkt der ersten Erhöhung in der Sequenz
        
        Args:
            sequence: Liste von Detection-Dictionaries
            
        Returns:
            Timestamp oder None
        """
        for i in range(1, len(sequence)):
            if sequence[i]['persons_detected'] > sequence[i-1]['persons_detected']:
                # FIX: Behandle None-Werte
                if (sequence[i].get('avg_confidence') or 0.0) >= self.min_confidence:
                    return sequence[i]['timestamp']
        return None
    
    def _find_first_decrease(self, sequence: List[Dict]) -> Optional[datetime]:
        """
        Findet Zeitpunkt der ersten Verringerung in der Sequenz
        
        Args:
            sequence: Liste von Detection-Dictionaries
            
        Returns:
            Timestamp oder None
        """
        for i in range(1, len(sequence)):
            if sequence[i]['persons_detected'] < sequence[i-1]['persons_detected']:
                # FIX: Behandle None-Werte
                if (sequence[i].get('avg_confidence') or 0.0) >= self.min_confidence:
                    return sequence[i]['timestamp']
        return None
    
    def _calculate_confidence(self, x_seq: List[Dict], y_seq: List[Dict], 
                             time_diff: float) -> float:
        """
        Berechnet Konfidenz der erkannten Bewegung
        
        Faktoren:
        - Durchschnittliche Detection-Konfidenz beider Kameras
        - Zeitliche Plausibilität (je näher beieinander, desto besser)
        - Konsistenz zwischen Kameras
        
        Args:
            x_seq: X-Sequenz
            y_seq: Y-Sequenz
            time_diff: Zeitdifferenz zwischen Ereignissen
            
        Returns:
            Konfidenz-Score (0.0 - 1.0)
        """
        # Durchschnittskonfidenzen - FIX: Behandle None-Werte
        x_confs = [(d.get('avg_confidence') or 0.0) for d in x_seq]
        y_confs = [(d.get('avg_confidence') or 0.0) for d in y_seq]
        
        x_avg = np.mean(x_confs) if x_confs else 0.0
        y_avg = np.mean(y_confs) if y_confs else 0.0
        
        avg_confidence = (x_avg + y_avg) / 2
        
        # Zeitdifferenz-Strafe
        # Je näher an 0, desto besser; max bei transition_window
        time_penalty = max(0.3, 1.0 - (time_diff / self.transition_window))
        
        # Konsistenz-Check
        x_delta = abs(self._calculate_delta(x_seq))
        y_delta = abs(self._calculate_delta(y_seq))
        
        # Perfekte Übereinstimmung = 1.0, sonst etwas weniger
        if x_delta == y_delta:
            consistency = 1.0
        elif x_delta == 0 or y_delta == 0:
            consistency = 0.5
        else:
            consistency = min(x_delta, y_delta) / max(x_delta, y_delta)
        
        # Kombiniere Faktoren
        final_confidence = avg_confidence * time_penalty * consistency
        
        return min(1.0, max(0.0, final_confidence))