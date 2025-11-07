#!/usr/bin/env python3
"""
ÜBERARBEITETE Bewegungserkennung für zwei separate Kameras
Die Kameras zeigen UNTERSCHIEDLICHE Bereiche (nicht dieselbe Person)

Konzept:
- Kamera X = AUßENbereich (vor der Tür)
- Kamera Y = INNENbereich (im Raum)
- Entry: Erst X sieht Person (außen), dann verschwindet sie aus X, dann Y sieht sie (innen)
- Exit: Erst Y sieht Person weniger (innen verlässt), dann X sieht Person (außen erscheint)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class MovementDetector:
    """
    Erkennt Bewegungen zwischen zwei SEPARATEN Kamerabereichen
    """
    
    def __init__(self, 
                 transition_window: float = 5.0,
                 min_confidence: float = 0.4):
        """
        Args:
            transition_window: Maximale Zeit zwischen Kamera-Ereignissen (Sekunden)
            min_confidence: Mindest-Konfidenz für Detections
        """
        self.transition_window = transition_window
        self.min_confidence = min_confidence
    
    def detect_movements(self, detections: List[Dict]) -> List[Dict]:
        """
        Analysiert Detections und findet Bewegungsmuster
        
        WICHTIG: Diese Methode erwartet, dass Kameras UNTERSCHIEDLICHE Bereiche zeigen!
        
        Args:
            detections: Liste von Detection-Dictionaries
            
        Returns:
            Liste von erkannten Bewegungen
        """
        if len(detections) < 3:  # Brauchen mindestens 3 für Übergangserkennung
            return []
        
        # Sortiere nach Zeit
        sorted_detections = sorted(detections, key=lambda x: x['timestamp'])
        
        # Separiere nach Quelle
        x_detections = [d for d in sorted_detections if d['source'] == 'input_x']
        y_detections = [d for d in sorted_detections if d['source'] == 'input_y']
        
        if len(x_detections) < 2 or len(y_detections) < 2:
            return []
        
        movements = []
        
        # Analysiere Entry-Muster: X↑ dann X↓ dann Y↑
        entry = self._detect_entry_pattern(x_detections, y_detections)
        if entry:
            movements.append(entry)
        
        # Analysiere Exit-Muster: Y↓ dann X↑
        exit_m = self._detect_exit_pattern(x_detections, y_detections)
        if exit_m:
            movements.append(exit_m)
        
        return movements
    
    def _detect_entry_pattern(self, x_seq: List[Dict], y_seq: List[Dict]) -> Optional[Dict]:
        """
        Entry-Muster für SEPARATE Kameras:
        1. X sieht Anstieg (Person erscheint außen)
        2. X sieht Abfall (Person verschwindet aus Außenbereich)
        3. Y sieht Anstieg (Person erscheint innen)
        
        Oder vereinfacht:
        - X zeigt temporären Peak
        - Y zeigt Anstieg kurz danach
        """
        # Suche nach Peak in X (Anstieg dann Abfall)
        x_peaks = self._find_peaks(x_seq)
        
        # Suche nach Anstiegen in Y
        y_increases = self._find_increases(y_seq)
        
        if not x_peaks or not y_increases:
            return None
        
        # Finde zeitlich passende Kombinationen
        for x_peak_time, x_peak_delta in x_peaks:
            for y_inc_time, y_inc_delta in y_increases:
                time_diff = (y_inc_time - x_peak_time).total_seconds()
                
                # Y-Anstieg muss NACH X-Peak kommen
                if 0 < time_diff < self.transition_window:
                    # Personenanzahl: Nehme Minimum für Konservativität
                    person_count = min(x_peak_delta, y_inc_delta)
                    
                    if person_count < 1:
                        continue
                    
                    confidence = self._calculate_confidence(
                        x_seq, y_seq, time_diff, x_peak_delta, y_inc_delta
                    )
                    
                    return {
                        'type': 'entry',
                        'person_count': person_count,
                        'confidence': confidence,
                        'time_diff': time_diff,
                        'x_delta': x_peak_delta,
                        'y_delta': y_inc_delta,
                        'sequence': {
                            'x_ids': [d['id'] for d in x_seq],
                            'y_ids': [d['id'] for d in y_seq]
                        },
                        'pattern': f'X peak ({x_peak_delta}) → Y increase ({y_inc_delta})'
                    }
        
        return None
    
    def _detect_exit_pattern(self, x_seq: List[Dict], y_seq: List[Dict]) -> Optional[Dict]:
        """
        Exit-Muster für SEPARATE Kameras:
        1. Y sieht Abfall (Person verlässt Innenbereich)
        2. X sieht Anstieg (Person erscheint außen)
        """
        # Suche nach Abfällen in Y
        y_decreases = self._find_decreases(y_seq)
        
        # Suche nach Anstiegen in X
        x_increases = self._find_increases(x_seq)
        
        if not y_decreases or not x_increases:
            return None
        
        # Finde zeitlich passende Kombinationen
        for y_dec_time, y_dec_delta in y_decreases:
            for x_inc_time, x_inc_delta in x_increases:
                time_diff = (x_inc_time - y_dec_time).total_seconds()
                
                # X-Anstieg muss NACH Y-Abfall kommen
                if 0 < time_diff < self.transition_window:
                    person_count = min(abs(y_dec_delta), x_inc_delta)
                    
                    if person_count < 1:
                        continue
                    
                    confidence = self._calculate_confidence(
                        x_seq, y_seq, time_diff, x_inc_delta, abs(y_dec_delta)
                    )
                    
                    return {
                        'type': 'exit',
                        'person_count': person_count,
                        'confidence': confidence,
                        'time_diff': time_diff,
                        'x_delta': x_inc_delta,
                        'y_delta': y_dec_delta,
                        'sequence': {
                            'x_ids': [d['id'] for d in x_seq],
                            'y_ids': [d['id'] for d in y_seq]
                        },
                        'pattern': f'Y decrease ({y_dec_delta}) → X increase ({x_inc_delta})'
                    }
        
        return None
    
    def _find_peaks(self, sequence: List[Dict]) -> List[Tuple[datetime, int]]:
        """
        Findet Peaks: Anstieg gefolgt von Abfall
        
        Returns:
            Liste von (Zeitpunkt, Delta) Tupeln
        """
        peaks = []
        
        if len(sequence) < 3:
            return peaks
        
        high_conf = [d for d in sequence 
                     if (d.get('avg_confidence') or 0.0) >= self.min_confidence]
        
        if len(high_conf) < 3:
            return peaks
        
        for i in range(1, len(high_conf) - 1):
            prev_count = high_conf[i-1]['persons_detected']
            curr_count = high_conf[i]['persons_detected']
            next_count = high_conf[i+1]['persons_detected']
            
            # Peak: vorher niedriger, danach niedriger
            if prev_count < curr_count > next_count:
                delta = curr_count - prev_count
                peaks.append((high_conf[i]['timestamp'], delta))
        
        return peaks
    
    def _find_increases(self, sequence: List[Dict]) -> List[Tuple[datetime, int]]:
        """
        Findet signifikante Anstiege
        
        Returns:
            Liste von (Zeitpunkt, Delta) Tupeln
        """
        increases = []
        
        high_conf = [d for d in sequence 
                     if (d.get('avg_confidence') or 0.0) >= self.min_confidence]
        
        if len(high_conf) < 2:
            return increases
        
        for i in range(1, len(high_conf)):
            prev_count = high_conf[i-1]['persons_detected']
            curr_count = high_conf[i]['persons_detected']
            
            if curr_count > prev_count:
                delta = curr_count - prev_count
                increases.append((high_conf[i]['timestamp'], delta))
        
        return increases
    
    def _find_decreases(self, sequence: List[Dict]) -> List[Tuple[datetime, int]]:
        """
        Findet signifikante Abfälle
        
        Returns:
            Liste von (Zeitpunkt, Delta) Tupeln (Delta ist negativ!)
        """
        decreases = []
        
        high_conf = [d for d in sequence 
                     if (d.get('avg_confidence') or 0.0) >= self.min_confidence]
        
        if len(high_conf) < 2:
            return decreases
        
        for i in range(1, len(high_conf)):
            prev_count = high_conf[i-1]['persons_detected']
            curr_count = high_conf[i]['persons_detected']
            
            if curr_count < prev_count:
                delta = curr_count - prev_count  # Negativ!
                decreases.append((high_conf[i]['timestamp'], delta))
        
        return decreases
    
    def _calculate_confidence(self, x_seq: List[Dict], y_seq: List[Dict], 
                             time_diff: float, x_delta: int, y_delta: int) -> float:
        """
        Berechnet Konfidenz der erkannten Bewegung
        """
        # Durchschnittskonfidenzen
        x_confs = [(d.get('avg_confidence') or 0.0) for d in x_seq]
        y_confs = [(d.get('avg_confidence') or 0.0) for d in y_seq]
        
        x_avg = np.mean(x_confs) if x_confs else 0.0
        y_avg = np.mean(y_confs) if y_confs else 0.0
        
        avg_confidence = (x_avg + y_avg) / 2
        
        # Zeitdifferenz-Strafe (je näher an 0, desto besser)
        time_penalty = max(0.3, 1.0 - (time_diff / self.transition_window))
        
        # Delta-Konsistenz (beide Kameras sollten ähnliche Änderungen sehen)
        if x_delta == 0 or y_delta == 0:
            consistency = 0.5
        elif x_delta == y_delta:
            consistency = 1.0
        else:
            consistency = min(x_delta, y_delta) / max(x_delta, y_delta)
        
        # Kombiniere Faktoren
        final_confidence = avg_confidence * time_penalty * consistency
        
        return min(1.0, max(0.0, final_confidence))