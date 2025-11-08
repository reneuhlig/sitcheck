#!/usr/bin/env python3
"""
VEREINFACHTER Movement Detector
Erkennt bereits kleinste Änderungen (ab 1 Person Differenz)
Mit robustem Fehlerhandling & ausführlichem Logging
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class MovementDetector:
    """
    Vereinfachter Movement Detector
    Reagiert bereits auf +1/-1 Person Änderung
    """
    
    def __init__(self, 
                 transition_window: float = 10.0,
                 min_confidence: float = 0.3):
        """
        Args:
            transition_window: Maximale Zeit zwischen Ereignissen (Sekunden)
            min_confidence: Mindest-Konfidenz (für Logging, keine Filterung)
        """
        self.transition_window = transition_window
        self.min_confidence = min_confidence
        
        print(f"   MovementDetector initialisiert:")
        print(f"   Transition Window: {transition_window}s")
        print(f"   Min Confidence: {min_confidence}")
    
    # =====================================================================================
    # Hauptfunktion
    # =====================================================================================
    
    def detect_movements(self, detections: List[Dict]) -> List[Dict]:
        """Analysiert Detections und findet Bewegungsmuster"""
        print(f"\n detect_movements() aufgerufen mit {len(detections)} Detections")
        
        if len(detections) < 2:
            print(f"Zu wenig Detections: {len(detections)} < 2")
            return []
        
        # Sortieren nach Zeit
        sorted_detections = sorted(detections, key=lambda x: x['timestamp'])
        print(f"  Zeitspanne: {sorted_detections[0]['timestamp']} bis {sorted_detections[-1]['timestamp']}")
        
        # Nach Quelle trennen
        x_detections = [d for d in sorted_detections if d.get('source') == 'input_x']
        y_detections = [d for d in sorted_detections if d.get('source') == 'input_y']
        
        print(f"  Input X: {len(x_detections)} Detections")
        print(f"  Input Y: {len(y_detections)} Detections")
        
        if len(x_detections) < 2 or len(y_detections) < 2:
            print(f"    Zu wenig Detections pro Quelle (min. 2 pro Quelle)")
            return []
        
        # Übersicht mit robustem Confidence-Handling
        self._print_detection_summary(x_detections, y_detections)
        
        movements = []
        
        # Entry
        print(f"\n     Suche Entry-Muster...")
        entry = self._detect_entry_pattern(x_detections, y_detections)
        if entry:
            movements.append(entry)
            print(f"   Entry erkannt: {entry['person_count']} Person(en)")
        else:
            print(f"   Kein Entry erkannt")
        
        # Exit
        print(f"\n   Suche Exit-Muster...")
        exit_m = self._detect_exit_pattern(x_detections, y_detections)
        if exit_m:
            movements.append(exit_m)
            print(f"   Exit erkannt: {exit_m['person_count']} Person(en)")
        else:
            print(f"   Kein Exit erkannt")
        
        print(f"\n Gesamt gefunden: {len(movements)} Bewegung(en)")
        return movements

    # =====================================================================================
    # Hilfsfunktionen
    # =====================================================================================
    
    def _print_detection_summary(self, x_seq: List[Dict], y_seq: List[Dict]):
        """Zeigt letzten Detections beider Quellen"""
        print(f"\n   Input X Detections:")
        for d in x_seq[-5:]:
            conf = d.get('avg_confidence')
            if conf is None or not isinstance(conf, (float, int)):
                conf = 0.0
            count = int(d.get('persons_detected', 0))
            ts = d.get('timestamp')
            time_str = ts.strftime('%H:%M:%S') if isinstance(ts, datetime) else "N/A"
            status = "OK" if conf >= self.min_confidence else "ERROR (zu niedrig)"
            print(f"      {time_str} | {count:2d} Pers. | Conf: {conf:.3f} {status}")
        
        print(f"\n   Input Y Detections:")
        for d in y_seq[-5:]:
            conf = d.get('avg_confidence')
            if conf is None or not isinstance(conf, (float, int)):
                conf = 0.0
            count = int(d.get('persons_detected', 0))
            ts = d.get('timestamp')
            time_str = ts.strftime('%H:%M:%S') if isinstance(ts, datetime) else "N/A"
            status = "OK" if conf >= self.min_confidence else "ERROR (zu niedrig)"
            print(f"      {time_str} | {count:2d} Pers. | Conf: {conf:.3f} {status}")
    
    # =====================================================================================
    # Mustererkennung Entry
    # =====================================================================================
    
    def _detect_entry_pattern(self, x_seq: List[Dict], y_seq: List[Dict]) -> Optional[Dict]:
        """Eintritt: Y steigt, X vorher aktiv"""
        print(f"      > Analysiere {len(y_seq)} Y-Detections auf Anstiege...")
        
        y_increases = self._find_any_increases(y_seq)
        print(f"      > Gefunden: {len(y_increases)} Anstiege in Y")
        
        if not y_increases:
            return None
        
        y_time, y_delta = y_increases[0]
        print(f"      > Bester Y-Anstieg: +{y_delta} um {y_time.strftime('%H:%M:%S')}")
        
        x_in_window = [d for d in x_seq if abs((d['timestamp'] - y_time).total_seconds()) < self.transition_window]
        if not x_in_window:
            print(f"      > Keine X-Detections im Zeitfenster")
            return None
        
        x_avg = np.mean([d.get('persons_detected', 0) for d in x_in_window])
        print(f"      > X-Durchschnitt: {x_avg:.1f}")
        
        person_count = max(1, min(y_delta, int(round(x_avg))))
        confidence = self._calculate_simple_confidence(x_seq, y_seq)
        
        return {
            'type': 'entry',
            'person_count': person_count,
            'confidence': confidence,
            'time_diff': 0.0,
            'x_delta': int(round(x_avg)),
            'y_delta': y_delta,
            'sequence': {'x_ids': [d['id'] for d in x_seq], 'y_ids': [d['id'] for d in y_seq]},
            'pattern': f'Y increase {y_delta}, X avg {x_avg:.1f}'
        }
    
    # =====================================================================================
    # Mustererkennung Exit
    # =====================================================================================
    
    def _detect_exit_pattern(self, x_seq: List[Dict], y_seq: List[Dict]) -> Optional[Dict]:
        """Austritt: Y fällt, X aktiv"""
        print(f"      > Analysiere {len(y_seq)} Y-Detections auf Abfälle...")
        
        y_decreases = self._find_any_decreases(y_seq)
        print(f"      > Gefunden: {len(y_decreases)} Abfälle in Y")
        
        if not y_decreases:
            return None
        
        y_time, y_delta = y_decreases[0]
        print(f"      > Bester Y-Abfall: {y_delta} um {y_time.strftime('%H:%M:%S')}")
        
        x_in_window = [d for d in x_seq if abs((d['timestamp'] - y_time).total_seconds()) < self.transition_window]
        if not x_in_window:
            print(f"      > Keine X-Detections im Zeitfenster")
            return None
        
        x_avg = np.mean([d.get('persons_detected', 0) for d in x_in_window])
        print(f"      > X-Durchschnitt: {x_avg:.1f}")
        
        person_count = max(1, min(abs(y_delta), int(round(x_avg))))
        confidence = self._calculate_simple_confidence(x_seq, y_seq)
        
        return {
            'type': 'exit',
            'person_count': person_count,
            'confidence': confidence,
            'time_diff': 0.0,
            'x_delta': int(round(x_avg)),
            'y_delta': y_delta,
            'sequence': {'x_ids': [d['id'] for d in x_seq], 'y_ids': [d['id'] for d in y_seq]},
            'pattern': f'Y decrease {y_delta}, X avg {x_avg:.1f}'
        }

    # =====================================================================================
    # Hilfslogik für Anstiege & Abfälle
    # =====================================================================================
    
    def _find_any_increases(self, sequence: List[Dict]) -> List[Tuple[datetime, int]]:
        """Finde ALLE Anstiege in der Y-Sequenz"""
        increases = []
        for i in range(1, len(sequence)):
            prev = int(sequence[i - 1].get('persons_detected', 0))
            curr = int(sequence[i].get('persons_detected', 0))
            if curr > prev:
                increases.append((sequence[i]['timestamp'], curr - prev))
                print(f"         Anstieg: +{curr - prev} um {sequence[i]['timestamp'].strftime('%H:%M:%S')}")
        return increases
    
    def _find_any_decreases(self, sequence: List[Dict]) -> List[Tuple[datetime, int]]:
        """Finde ALLE Abfälle in der Y-Sequenz"""
        decreases = []
        for i in range(1, len(sequence)):
            prev = int(sequence[i - 1].get('persons_detected', 0))
            curr = int(sequence[i].get('persons_detected', 0))
            if curr < prev:
                decreases.append((sequence[i]['timestamp'], curr - prev))
                print(f"         Abfall: {curr - prev} um {sequence[i]['timestamp'].strftime('%H:%M:%S')}")
        return decreases
    
    # =====================================================================================
    # Confidence
    # =====================================================================================
    
    def _calculate_simple_confidence(self, x_seq: List[Dict], y_seq: List[Dict]) -> float:
        """Berechne vereinfachte Konfidenz"""
        x_confs = [d.get('avg_confidence') or 0.5 for d in x_seq]
        y_confs = [d.get('avg_confidence') or 0.5 for d in y_seq]
        avg_conf = (np.mean(x_confs) + np.mean(y_confs)) / 2
        return round(min(1.0, max(0.4, avg_conf * 0.8)), 3)


# =====================================================================================
# TESTLAUF
# =====================================================================================

if __name__ == "__main__":
    from datetime import timedelta
    base_time = datetime.now()
    
    test_detections = [
        {'id': 1, 'timestamp': base_time, 'source': 'input_x', 'persons_detected': 5, 'avg_confidence': None},
        {'id': 2, 'timestamp': base_time + timedelta(seconds=3), 'source': 'input_x', 'persons_detected': 6, 'avg_confidence': 0.7},
        {'id': 3, 'timestamp': base_time + timedelta(seconds=6), 'source': 'input_y', 'persons_detected': 1, 'avg_confidence': None},
        {'id': 4, 'timestamp': base_time + timedelta(seconds=9), 'source': 'input_y', 'persons_detected': 2, 'avg_confidence': 0.8},
        {'id': 5, 'timestamp': base_time + timedelta(seconds=12), 'source': 'input_y', 'persons_detected': 2, 'avg_confidence': 0.75},
    ]
    
    detector = MovementDetector()
    moves = detector.detect_movements(test_detections)
    print("\nErkannte Bewegungen:")
    for m in moves:
        print(f"  {m['type']} ({m['person_count']} Pers, conf={m['confidence']})")
