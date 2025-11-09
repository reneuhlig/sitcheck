#!/usr/bin/env python3
"""
VERBESSERTER Movement Detector
- Verhindert Doppelerkennungen durch ID-Tracking
- Trennung: Detection-Confidence vs. Pattern-Confidence
- Zyklusübergreifende Mustererkennung
- Sequentielle Verarbeitung mehrerer Muster
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class MovementDetector:
    """
    Verbesserter Movement Detector mit robuster Mustererkennung
    """
    
    def __init__(self, 
                 transition_window: float = 10.0,
                 min_pattern_confidence: float = 0.3,
                 max_patterns_per_type: int = 3):
        """
        Args:
            transition_window: Maximale Zeit zwischen Ereignissen (Sekunden)
            min_pattern_confidence: Mindest-Konfidenz für Pattern (nicht Detection)
            max_patterns_per_type: Maximale Anzahl Entry/Exit pro Zyklus
        """
        self.transition_window = transition_window
        self.min_pattern_confidence = min_pattern_confidence
        self.max_patterns_per_type = max_patterns_per_type
        
        # Tracking verwendeter IDs innerhalb eines Durchlaufs
        self.used_ids_this_cycle = set()
        
        print(f"   MovementDetector initialisiert:")
        print(f"   Transition Window: {transition_window}s")
        print(f"   Min Pattern Confidence: {min_pattern_confidence}")
        print(f"   Max Patterns per Type: {max_patterns_per_type}")
    
    def detect_movements(self, detections: List[Dict]) -> List[Dict]:
        """
        Analysiert Detections und findet Bewegungsmuster
        
        Args:
            detections: Liste aller verfügbaren Detections (auch bereits verarbeitete)
            
        Returns:
            Liste von erkannten Bewegungen
        """
        print(f"\n[MOVEMENT] detect_movements() mit {len(detections)} Detections")
        
        if len(detections) < 2:
            print(f"[MOVEMENT] Zu wenig Detections: {len(detections)} < 2")
            return []
        
        # Filtere bereits in Mustern verwendete Detections aus
        available = [d for d in detections if not d.get('used_in_pattern', False)]
        print(f"[MOVEMENT] Verfügbare Detections: {len(available)}/{len(detections)}")
        
        if len(available) < 2:
            print(f"[MOVEMENT] Zu wenig verfügbare Detections")
            return []
        
        # Sortieren nach Zeit
        sorted_detections = sorted(available, key=lambda x: x['timestamp'])
        print(f"[MOVEMENT] Zeitspanne: {sorted_detections[0]['timestamp']} bis {sorted_detections[-1]['timestamp']}")
        
        # Nach Quelle trennen
        x_detections = [d for d in sorted_detections if d.get('source') == 'input_x']
        y_detections = [d for d in sorted_detections if d.get('source') == 'input_y']
        
        print(f"[MOVEMENT] Input X: {len(x_detections)} Detections")
        print(f"[MOVEMENT] Input Y: {len(y_detections)} Detections")
        
        if len(x_detections) < 1 or len(y_detections) < 2:
            print(f"[MOVEMENT] Zu wenig Detections pro Quelle (min. X=1, Y=2)")
            return []
        
        # Zeige Detection-Übersicht
        self._print_detection_summary(x_detections, y_detections)
        
        # Reset ID-Tracking für diesen Zyklus
        self.used_ids_this_cycle = set()
        
        movements = []
        
        # Entry-Muster (Priorität 1)
        print(f"\n[MOVEMENT] === SUCHE ENTRY-MUSTER ===")
        entry_movements = self._detect_multiple_entries(x_detections, y_detections)
        movements.extend(entry_movements)
        print(f"[MOVEMENT] Entry-Muster gefunden: {len(entry_movements)}")
        
        # Exit-Muster (Priorität 2)
        print(f"\n[MOVEMENT] === SUCHE EXIT-MUSTER ===")
        exit_movements = self._detect_multiple_exits(x_detections, y_detections)
        movements.extend(exit_movements)
        print(f"[MOVEMENT] Exit-Muster gefunden: {len(exit_movements)}")
        
        print(f"\n[MOVEMENT] Gesamt gefunden: {len(movements)} Bewegung(en)")
        return movements
    
    def _detect_multiple_entries(self, x_seq: List[Dict], y_seq: List[Dict]) -> List[Dict]:
        """
        Findet mehrere Entry-Muster sequentiell
        
        Returns:
            Liste von Entry-Bewegungen
        """
        entries = []
        remaining_x = x_seq.copy()
        remaining_y = y_seq.copy()
        
        for attempt in range(self.max_patterns_per_type):
            # Filtere bereits verwendete IDs
            available_x = [d for d in remaining_x if d['id'] not in self.used_ids_this_cycle]
            available_y = [d for d in remaining_y if d['id'] not in self.used_ids_this_cycle]
            
            if len(available_x) < 1 or len(available_y) < 2:
                break
            
            print(f"[MOVEMENT]   Entry-Versuch {attempt + 1}: X={len(available_x)}, Y={len(available_y)} verfügbar")
            
            entry = self._detect_entry_pattern(available_x, available_y)
            
            if entry:
                # Markiere verwendete IDs
                used_ids = entry['sequence']['x_ids'] + entry['sequence']['y_ids']
                self.used_ids_this_cycle.update(used_ids)
                entries.append(entry)
                print(f"[MOVEMENT]   Entry erkannt: {entry['person_count']} Person(en), Conf={entry['confidence']:.2f}")
            else:
                break
        
        return entries
    
    def _detect_multiple_exits(self, x_seq: List[Dict], y_seq: List[Dict]) -> List[Dict]:
        """
        Findet mehrere Exit-Muster sequentiell
        
        Returns:
            Liste von Exit-Bewegungen
        """
        exits = []
        remaining_x = x_seq.copy()
        remaining_y = y_seq.copy()
        
        for attempt in range(self.max_patterns_per_type):
            # Filtere bereits verwendete IDs
            available_x = [d for d in remaining_x if d['id'] not in self.used_ids_this_cycle]
            available_y = [d for d in remaining_y if d['id'] not in self.used_ids_this_cycle]
            
            if len(available_x) < 1 or len(available_y) < 2:
                break
            
            print(f"[MOVEMENT]   Exit-Versuch {attempt + 1}: X={len(available_x)}, Y={len(available_y)} verfügbar")
            
            exit_m = self._detect_exit_pattern(available_x, available_y)
            
            if exit_m:
                # Markiere verwendete IDs
                used_ids = exit_m['sequence']['x_ids'] + exit_m['sequence']['y_ids']
                self.used_ids_this_cycle.update(used_ids)
                exits.append(exit_m)
                print(f"[MOVEMENT]   Exit erkannt: {exit_m['person_count']} Person(en), Conf={exit_m['confidence']:.2f}")
            else:
                break
        
        return exits
    
    def _detect_entry_pattern(self, x_seq: List[Dict], y_seq: List[Dict]) -> Optional[Dict]:
        """
        Eintritt: Y steigt, X vorher/währenddessen aktiv
        
        Returns:
            Movement-Dict oder None
        """
        print(f"[MOVEMENT]     Analysiere {len(y_seq)} Y-Detections auf Anstiege...")
        
        # Finde alle Anstiege in Y
        y_increases = self._find_all_increases(y_seq)
        
        if not y_increases:
            print(f"[MOVEMENT]     Keine Y-Anstiege gefunden")
            return None
        
        print(f"[MOVEMENT]     {len(y_increases)} Y-Anstiege gefunden")
        
        # Sortiere nach Stärke (größtes Delta zuerst)
        y_increases.sort(key=lambda x: x[1], reverse=True)
        
        # Nimm stärksten Anstieg
        y_time, y_delta, y_idx = y_increases[0]
        print(f"[MOVEMENT]     Stärkster Y-Anstieg: +{y_delta} um {y_time.strftime('%H:%M:%S')}")
        
        # Finde X-Detections im Zeitfenster
        x_in_window = [d for d in x_seq 
                       if abs((d['timestamp'] - y_time).total_seconds()) < self.transition_window]
        
        if not x_in_window:
            print(f"[MOVEMENT]     Keine X-Detections im Zeitfenster")
            return None
        
        x_avg = np.mean([d.get('persons_detected', 0) for d in x_in_window])
        print(f"[MOVEMENT]     X-Durchschnitt im Fenster: {x_avg:.1f}")
        
        # Bestimme Personenanzahl (konservativ)
        person_count = max(1, min(y_delta, int(round(x_avg))))
        
        # Berechne Pattern Confidence
        confidence = self._calculate_pattern_confidence(
            x_seq=x_in_window,
            y_seq=y_seq[max(0, y_idx-2):y_idx+2],
            delta_y=y_delta,
            pattern_type='entry'
        )
        
        print(f"[MOVEMENT]     Pattern Confidence: {confidence:.2f}")
        
        # Prüfe Schwellwert
        if confidence < self.min_pattern_confidence:
            print(f"[MOVEMENT]     Verworfen: Confidence zu niedrig ({confidence:.2f} < {self.min_pattern_confidence})")
            return None
        
        # Sammle verwendete IDs
        x_ids = [d['id'] for d in x_in_window]
        y_ids = [d['id'] for d in y_seq[max(0, y_idx-1):min(len(y_seq), y_idx+2)]]
        
        return {
            'type': 'entry',
            'person_count': person_count,
            'confidence': confidence,
            'time_diff': 0.0,
            'x_delta': int(round(x_avg)),
            'y_delta': y_delta,
            'sequence': {'x_ids': x_ids, 'y_ids': y_ids},
            'pattern': f'Y increase +{y_delta}, X avg {x_avg:.1f}'
        }
    
    def _detect_exit_pattern(self, x_seq: List[Dict], y_seq: List[Dict]) -> Optional[Dict]:
        """
        Austritt: Y fällt, X aktiv
        
        Returns:
            Movement-Dict oder None
        """
        print(f"[MOVEMENT]     Analysiere {len(y_seq)} Y-Detections auf Abfälle...")
        
        # Finde alle Abfälle in Y
        y_decreases = self._find_all_decreases(y_seq)
        
        if not y_decreases:
            print(f"[MOVEMENT]     Keine Y-Abfälle gefunden")
            return None
        
        print(f"[MOVEMENT]     {len(y_decreases)} Y-Abfälle gefunden")
        
        # Sortiere nach Stärke (größtes absolutes Delta zuerst)
        y_decreases.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Nimm stärksten Abfall
        y_time, y_delta, y_idx = y_decreases[0]
        print(f"[MOVEMENT]     Stärkster Y-Abfall: {y_delta} um {y_time.strftime('%H:%M:%S')}")
        
        # Finde X-Detections im Zeitfenster
        x_in_window = [d for d in x_seq 
                       if abs((d['timestamp'] - y_time).total_seconds()) < self.transition_window]
        
        if not x_in_window:
            print(f"[MOVEMENT]     Keine X-Detections im Zeitfenster")
            return None
        
        x_avg = np.mean([d.get('persons_detected', 0) for d in x_in_window])
        print(f"[MOVEMENT]     X-Durchschnitt im Fenster: {x_avg:.1f}")
        
        # Bestimme Personenanzahl (konservativ)
        person_count = max(1, min(abs(y_delta), int(round(x_avg))))
        
        # Berechne Pattern Confidence
        confidence = self._calculate_pattern_confidence(
            x_seq=x_in_window,
            y_seq=y_seq[max(0, y_idx-2):y_idx+2],
            delta_y=abs(y_delta),
            pattern_type='exit'
        )
        
        print(f"[MOVEMENT]     Pattern Confidence: {confidence:.2f}")
        
        # Prüfe Schwellwert
        if confidence < self.min_pattern_confidence:
            print(f"[MOVEMENT]     Verworfen: Confidence zu niedrig ({confidence:.2f} < {self.min_pattern_confidence})")
            return None
        
        # Sammle verwendete IDs
        x_ids = [d['id'] for d in x_in_window]
        y_ids = [d['id'] for d in y_seq[max(0, y_idx-1):min(len(y_seq), y_idx+2)]]
        
        return {
            'type': 'exit',
            'person_count': person_count,
            'confidence': confidence,
            'time_diff': 0.0,
            'x_delta': int(round(x_avg)),
            'y_delta': y_delta,
            'sequence': {'x_ids': x_ids, 'y_ids': y_ids},
            'pattern': f'Y decrease {y_delta}, X avg {x_avg:.1f}'
        }
    
    def _find_all_increases(self, sequence: List[Dict]) -> List[Tuple[datetime, int, int]]:
        """
        Finde alle Anstiege in der Sequenz
        
        Returns:
            Liste von (timestamp, delta, index)
        """
        increases = []
        for i in range(1, len(sequence)):
            prev = int(sequence[i - 1].get('persons_detected', 0))
            curr = int(sequence[i].get('persons_detected', 0))
            if curr > prev:
                increases.append((sequence[i]['timestamp'], curr - prev, i))
        return increases
    
    def _find_all_decreases(self, sequence: List[Dict]) -> List[Tuple[datetime, int, int]]:
        """
        Finde alle Abfälle in der Sequenz
        
        Returns:
            Liste von (timestamp, delta, index)
        """
        decreases = []
        for i in range(1, len(sequence)):
            prev = int(sequence[i - 1].get('persons_detected', 0))
            curr = int(sequence[i].get('persons_detected', 0))
            if curr < prev:
                decreases.append((sequence[i]['timestamp'], curr - prev, i))
        return decreases
    
    def _calculate_pattern_confidence(self, x_seq: List[Dict], y_seq: List[Dict], 
                                      delta_y: int, pattern_type: str) -> float:
        """
        Berechnet Pattern Confidence basierend auf Musterstärke
        
        Komponenten:
        - Pattern Strength (40%): Wie stark ist die Änderung?
        - Temporal Coherence (30%): Zeitliche Konsistenz
        - Min Detection Quality (30%): Schlechteste Detection im Muster
        
        Args:
            x_seq: X-Detections im Muster
            y_seq: Y-Detections im Muster
            delta_y: Änderung in Y
            pattern_type: 'entry' oder 'exit'
            
        Returns:
            Confidence-Wert zwischen 0.0 und 1.0
        """
        # 1. Pattern Strength: Größe der Änderung
        # 1 Person = 0.33, 2 = 0.67, 3+ = 1.0
        pattern_strength = min(1.0, abs(delta_y) / 3.0)
        
        # 2. Temporal Coherence: Zeitliche Konsistenz
        if len(y_seq) > 1:
            time_span = (y_seq[-1]['timestamp'] - y_seq[0]['timestamp']).total_seconds()
            # Optimal: <= 10s, danach Abfall
            temporal_score = 1.0 if time_span <= 10 else max(0.5, 10.0 / time_span)
        else:
            temporal_score = 0.8
        
        # 3. Min Detection Quality: Schlechteste Detection
        # Verhindert, dass völlig unsichere Detections verwendet werden
        all_confs = []
        for d in x_seq + y_seq:
            conf = d.get('avg_confidence')
            if conf is None or not isinstance(conf, (float, int)):
                conf = 0.5  # Default für fehlende Werte
            all_confs.append(conf)
        
        min_detection_conf = min(all_confs) if all_confs else 0.5
        
        # 4. X-Konsistenz: Stabilität in X (nur bei Entry/Exit relevant)
        if len(x_seq) > 1:
            x_values = [d.get('persons_detected', 0) for d in x_seq]
            x_std = np.std(x_values)
            x_mean = np.mean(x_values)
            # Je stabiler X, desto besser (niedrige Varianz = gut)
            x_stability = 1.0 if x_mean == 0 else max(0.5, 1.0 - (x_std / max(1.0, x_mean)))
        else:
            x_stability = 0.8
        
        # Gewichtete Kombination
        confidence = (
            0.35 * pattern_strength +
            0.25 * temporal_score +
            0.25 * min_detection_conf +
            0.15 * x_stability
        )
        
        return round(max(0.1, min(1.0, confidence)), 3)
    
    def _print_detection_summary(self, x_seq: List[Dict], y_seq: List[Dict]):
        """Zeigt Übersicht der Detections mit Original-Confidence"""
        print(f"\n[MOVEMENT]   === Input X Detections (letzte 5) ===")
        for d in x_seq[-5:]:
            conf = d.get('avg_confidence')
            if conf is None or not isinstance(conf, (float, int)):
                conf_str = "N/A"
            else:
                conf_str = f"{conf:.3f}"
            
            count = int(d.get('persons_detected', 0))
            ts = d.get('timestamp')
            time_str = ts.strftime('%H:%M:%S') if isinstance(ts, datetime) else "N/A"
            used = "USED" if d.get('used_in_pattern', False) else "FREE"
            
            print(f"[MOVEMENT]     ID={d['id']:4d} | {time_str} | {count:2d} Pers | Conf={conf_str:>5s} | {used}")
        
        print(f"\n[MOVEMENT]   === Input Y Detections (letzte 5) ===")
        for d in y_seq[-5:]:
            conf = d.get('avg_confidence')
            if conf is None or not isinstance(conf, (float, int)):
                conf_str = "N/A"
            else:
                conf_str = f"{conf:.3f}"
            
            count = int(d.get('persons_detected', 0))
            ts = d.get('timestamp')
            time_str = ts.strftime('%H:%M:%S') if isinstance(ts, datetime) else "N/A"
            used = "USED" if d.get('used_in_pattern', False) else "FREE"
            
            print(f"[MOVEMENT]     ID={d['id']:4d} | {time_str} | {count:2d} Pers | Conf={conf_str:>5s} | {used}")


if __name__ == "__main__":
    from datetime import timedelta
    base_time = datetime.now()
    
    test_detections = [
        {'id': 1, 'timestamp': base_time, 'source': 'input_x', 'persons_detected': 5, 'avg_confidence': 0.8, 'used_in_pattern': False},
        {'id': 2, 'timestamp': base_time + timedelta(seconds=3), 'source': 'input_x', 'persons_detected': 6, 'avg_confidence': 0.7, 'used_in_pattern': False},
        {'id': 3, 'timestamp': base_time + timedelta(seconds=6), 'source': 'input_y', 'persons_detected': 1, 'avg_confidence': 0.75, 'used_in_pattern': False},
        {'id': 4, 'timestamp': base_time + timedelta(seconds=9), 'source': 'input_y', 'persons_detected': 2, 'avg_confidence': 0.8, 'used_in_pattern': False},
        {'id': 5, 'timestamp': base_time + timedelta(seconds=12), 'source': 'input_y', 'persons_detected': 2, 'avg_confidence': 0.75, 'used_in_pattern': False},
    ]
    
    detector = MovementDetector()
    moves = detector.detect_movements(test_detections)
    print("\nErkannte Bewegungen:")
    for m in moves:
        print(f"  {m['type']} ({m['person_count']} Pers, conf={m['confidence']:.2f})")