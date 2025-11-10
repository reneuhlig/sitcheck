#!/usr/bin/env python3
"""
SEQUENZ-BASIERTER Movement Detector mit Richtungslogik

HAUPTVERBESSERUNG:
- Entry: Y→X Sequenz (Person erscheint ZUERST in Y, DANN in X)
- Exit:  X→Y Sequenz (Person erscheint ZUERST in X, DANN in Y)

Verhindert Falsch-Positive durch zeitliche Reihenfolgen-Prüfung
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class MovementDetector:
    """
    Sequenz-basierter Movement Detector mit Richtungslogik
    """
    
    def __init__(self, 
                 transition_window: float = 10.0,
                 min_pattern_confidence: float = 0.3,
                 max_patterns_per_type: int = 3):
        """
        Args:
            transition_window: Maximale Zeit zwischen X und Y Events (Sekunden)
            min_pattern_confidence: Mindest-Konfidenz für Pattern
            max_patterns_per_type: Maximale Anzahl Entry/Exit pro Zyklus
        """
        self.transition_window = transition_window
        self.min_pattern_confidence = min_pattern_confidence
        self.max_patterns_per_type = max_patterns_per_type
        
        # Tracking verwendeter IDs
        self.used_ids_this_cycle = set()
        
        print(f"   MovementDetector initialisiert:")
        print(f"   Transition Window: {transition_window}s")
        print(f"   Min Pattern Confidence: {min_pattern_confidence}")
        print(f"   Max Patterns per Type: {max_patterns_per_type}")
    
    def detect_movements(self, detections: List[Dict]) -> List[Dict]:
        """
        Analysiert Detections und findet Bewegungsmuster
        
        Args:
            detections: Liste aller verfügbaren Detections
            
        Returns:
            Liste von erkannten Bewegungen
        """
        print(f"\n[MOVEMENT] detect_movements() mit {len(detections)} Detections")
        
        if len(detections) < 2:
            print(f"[MOVEMENT] Zu wenig Detections: {len(detections)} < 2")
            return []
        
        # Filtere bereits verwendete Detections
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
        
        if len(x_detections) < 1 or len(y_detections) < 1:
            print(f"[MOVEMENT] Zu wenig Detections pro Quelle (min. X=1, Y=1)")
            return []
        
        # Zeige Detection-Übersicht
        self._print_detection_summary(x_detections, y_detections)
        
        # Reset ID-Tracking
        self.used_ids_this_cycle = set()
        
        movements = []
        
        # Entry-Muster: Y→X Sequenz (Priorität 1)
        print(f"\n[MOVEMENT] === SUCHE ENTRY-MUSTER (Y→X Sequenz) ===")
        entry_movements = self._detect_multiple_entries(x_detections, y_detections)
        movements.extend(entry_movements)
        print(f"[MOVEMENT] Entry-Muster gefunden: {len(entry_movements)}")
        
        # Exit-Muster: X→Y Sequenz (Priorität 2)
        print(f"\n[MOVEMENT] === SUCHE EXIT-MUSTER (X→Y Sequenz) ===")
        exit_movements = self._detect_multiple_exits(x_detections, y_detections)
        movements.extend(exit_movements)
        print(f"[MOVEMENT] Exit-Muster gefunden: {len(exit_movements)}")
        
        print(f"\n[MOVEMENT] Gesamt gefunden: {len(movements)} Bewegung(en)")
        return movements
    
    def _detect_multiple_entries(self, x_seq: List[Dict], y_seq: List[Dict]) -> List[Dict]:
        """
        Findet mehrere Entry-Muster sequentiell
        Entry = Y→X Sequenz (Person erscheint ZUERST in Y, DANN in X)
        
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
            
            if len(available_x) < 1 or len(available_y) < 1:
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
        Exit = X→Y Sequenz (Person erscheint ZUERST in X, DANN in Y)
        
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
            
            if len(available_x) < 1 or len(available_y) < 1:
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
        Entry: Y→X Sequenz
        Person erscheint ZUERST in Y (0→1), DANN wird X aktiv
        
        Returns:
            Movement-Dict oder None
        """
        print(f"[MOVEMENT]     Analysiere Y→X Sequenzen für Entry...")
        
        # Finde alle Y-Anstiege (0→1 oder Anstieg)
        y_increases = self._find_all_increases(y_seq)
        
        if not y_increases:
            print(f"[MOVEMENT]     Keine Y-Anstiege gefunden")
            return None
        
        print(f"[MOVEMENT]     {len(y_increases)} Y-Anstiege gefunden")
        
        # Sortiere nach Stärke (größtes Delta zuerst)
        y_increases.sort(key=lambda x: x[1], reverse=True)
        
        # Prüfe jeden Y-Anstieg auf nachfolgende X-Aktivität
        for y_time, y_delta, y_idx in y_increases:
            print(f"[MOVEMENT]     Prüfe Y-Anstieg: +{y_delta} um {y_time.strftime('%H:%M:%S')}")
            
            # Suche X-Aktivität NACH diesem Y-Anstieg
            x_after_y = [d for d in x_seq 
                         if d['timestamp'] > y_time 
                         and (d['timestamp'] - y_time).total_seconds() < self.transition_window]
            
            if not x_after_y:
                print(f"[MOVEMENT]     Keine X-Aktivität NACH Y-Anstieg im Zeitfenster")
                continue
            
            # Prüfe ob X Person-Aktivität zeigt
            x_person_count = sum(d.get('persons_detected', 0) for d in x_after_y)
            
            if x_person_count == 0:
                print(f"[MOVEMENT]     Keine Personen in X NACH Y-Anstieg")
                continue
            
            # Berechne X-Durchschnitt
            x_avg = np.mean([d.get('persons_detected', 0) for d in x_after_y])
            print(f"[MOVEMENT]     ✓ X-Aktivität NACH Y: Durchschnitt {x_avg:.1f} Personen")
            
            # Bestimme Personenanzahl (konservativ)
            person_count = max(1, min(y_delta, int(round(x_avg))))
            
            # Sammle Y-Detections für Pattern (inkl. vorherige 0-Werte)
            y_pattern_detections = y_seq[max(0, y_idx-2):min(len(y_seq), y_idx+2)]
            
            # Berechne Pattern Confidence
            confidence = self._calculate_pattern_confidence(
                x_seq=x_after_y,
                y_seq=y_pattern_detections,
                delta=y_delta,
                pattern_type='entry'
            )
            
            print(f"[MOVEMENT]     Pattern Confidence: {confidence:.2f}")
            
            # Prüfe Schwellwert
            if confidence < self.min_pattern_confidence:
                print(f"[MOVEMENT]     Verworfen: Confidence zu niedrig ({confidence:.2f} < {self.min_pattern_confidence})")
                continue
            
            # ERFOLG: Entry-Muster gefunden
            x_ids = [d['id'] for d in x_after_y]
            y_ids = [d['id'] for d in y_pattern_detections]
            
            return {
                'type': 'entry',
                'person_count': person_count,
                'confidence': confidence,
                'time_diff': (x_after_y[0]['timestamp'] - y_time).total_seconds(),
                'x_delta': int(round(x_avg)),
                'y_delta': y_delta,
                'sequence': {'x_ids': x_ids, 'y_ids': y_ids},
                'pattern': f'Y→X: Y increase +{y_delta} @ {y_time.strftime("%H:%M:%S")}, then X avg {x_avg:.1f}'
            }
        
        print(f"[MOVEMENT]     Kein gültiges Entry-Muster gefunden")
        return None
    
    def _detect_exit_pattern(self, x_seq: List[Dict], y_seq: List[Dict]) -> Optional[Dict]:
        """
        Exit: X→Y Sequenz
        Person erscheint ZUERST in X (0→1), DANN wird Y aktiv
        
        Returns:
            Movement-Dict oder None
        """
        print(f"[MOVEMENT]     Analysiere X→Y Sequenzen für Exit...")
        
        # Finde alle X-Anstiege (0→1 oder Anstieg)
        x_increases = self._find_all_increases(x_seq)
        
        if not x_increases:
            print(f"[MOVEMENT]     Keine X-Anstiege gefunden")
            return None
        
        print(f"[MOVEMENT]     {len(x_increases)} X-Anstiege gefunden")
        
        # Sortiere nach Stärke
        x_increases.sort(key=lambda x: x[1], reverse=True)
        
        # Prüfe jeden X-Anstieg auf nachfolgende Y-Aktivität
        for x_time, x_delta, x_idx in x_increases:
            print(f"[MOVEMENT]     Prüfe X-Anstieg: +{x_delta} um {x_time.strftime('%H:%M:%S')}")
            
            # Suche Y-Aktivität NACH diesem X-Anstieg
            y_after_x = [d for d in y_seq 
                         if d['timestamp'] > x_time 
                         and (d['timestamp'] - x_time).total_seconds() < self.transition_window]
            
            if not y_after_x:
                print(f"[MOVEMENT]     Keine Y-Aktivität NACH X-Anstieg im Zeitfenster")
                continue
            
            # Prüfe ob Y Person-Aktivität zeigt
            y_person_count = sum(d.get('persons_detected', 0) for d in y_after_x)
            
            if y_person_count == 0:
                print(f"[MOVEMENT]     Keine Personen in Y NACH X-Anstieg")
                continue
            
            # Berechne Y-Durchschnitt
            y_avg = np.mean([d.get('persons_detected', 0) for d in y_after_x])
            print(f"[MOVEMENT]     ✓ Y-Aktivität NACH X: Durchschnitt {y_avg:.1f} Personen")
            
            # Bestimme Personenanzahl (konservativ)
            person_count = max(1, min(x_delta, int(round(y_avg))))
            
            # Sammle X-Detections für Pattern
            x_pattern_detections = x_seq[max(0, x_idx-2):min(len(x_seq), x_idx+2)]
            
            # Berechne Pattern Confidence
            confidence = self._calculate_pattern_confidence(
                x_seq=x_pattern_detections,
                y_seq=y_after_x,
                delta=x_delta,
                pattern_type='exit'
            )
            
            print(f"[MOVEMENT]     Pattern Confidence: {confidence:.2f}")
            
            # Prüfe Schwellwert
            if confidence < self.min_pattern_confidence:
                print(f"[MOVEMENT]     Verworfen: Confidence zu niedrig ({confidence:.2f} < {self.min_pattern_confidence})")
                continue
            
            # ERFOLG: Exit-Muster gefunden
            x_ids = [d['id'] for d in x_pattern_detections]
            y_ids = [d['id'] for d in y_after_x]
            
            return {
                'type': 'exit',
                'person_count': person_count,
                'confidence': confidence,
                'time_diff': (y_after_x[0]['timestamp'] - x_time).total_seconds(),
                'x_delta': x_delta,
                'y_delta': int(round(y_avg)),
                'sequence': {'x_ids': x_ids, 'y_ids': y_ids},
                'pattern': f'X→Y: X increase +{x_delta} @ {x_time.strftime("%H:%M:%S")}, then Y avg {y_avg:.1f}'
            }
        
        print(f"[MOVEMENT]     Kein gültiges Exit-Muster gefunden")
        return None
    
    def _find_all_increases(self, sequence: List[Dict]) -> List[Tuple[datetime, int, int]]:
        """
        Finde alle Anstiege in der Sequenz (0→1 oder N→N+1)
        
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
        Finde alle Abfälle in der Sequenz (N→N-1 oder 1→0)
        
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
                                      delta: int, pattern_type: str) -> float:
        """
        Berechnet Pattern Confidence basierend auf Musterstärke
        
        Komponenten:
        - Pattern Strength (35%): Wie stark ist die Änderung?
        - Temporal Coherence (30%): Zeitliche Konsistenz der Sequenz
        - Detection Quality (25%): Qualität der Detections
        - Sequence Clarity (10%): Wie klar ist die Sequenz?
        
        Args:
            x_seq: X-Detections im Muster
            y_seq: Y-Detections im Muster
            delta: Änderung (Personen)
            pattern_type: 'entry' oder 'exit'
            
        Returns:
            Confidence-Wert zwischen 0.0 und 1.0
        """
        # 1. Pattern Strength: Größe der Änderung
        # 1 Person = 0.4, 2 = 0.7, 3+ = 1.0
        pattern_strength = min(1.0, 0.4 + (delta - 1) * 0.3)
        
        # 2. Temporal Coherence: Zeitliche Konsistenz
        all_detections = x_seq + y_seq
        if len(all_detections) > 1:
            sorted_dets = sorted(all_detections, key=lambda d: d['timestamp'])
            time_span = (sorted_dets[-1]['timestamp'] - sorted_dets[0]['timestamp']).total_seconds()
            # Optimal: <= 10s = 1.0, linear fallend bis 20s = 0.5
            if time_span <= 10:
                temporal_score = 1.0
            elif time_span <= 20:
                temporal_score = 1.0 - ((time_span - 10) / 10) * 0.5
            else:
                temporal_score = max(0.3, 0.5 - ((time_span - 20) / 30) * 0.2)
        else:
            temporal_score = 0.7
        
        # 3. Detection Quality: Durchschnittliche Confidence
        all_confs = []
        for d in x_seq + y_seq:
            conf = d.get('avg_confidence')
            if conf is None or not isinstance(conf, (float, int)) or conf == 0:
                conf = 0.5  # Default für 0-Personen Detections
            all_confs.append(conf)
        
        avg_detection_conf = np.mean(all_confs) if all_confs else 0.5
        
        # 4. Sequence Clarity: Sind die Übergänge klar?
        # Hohe Klarheit = wenig Rauschen in der Sequenz
        clarity_score = 0.8  # Base score
        
        # Bonus für klare 0→1 Übergänge
        if pattern_type == 'entry':
            # Prüfe Y: sollte von 0 zu >0 gehen
            y_values = [d.get('persons_detected', 0) for d in y_seq]
            if len(y_values) >= 2 and y_values[0] == 0 and max(y_values) > 0:
                clarity_score = 1.0
        else:  # exit
            # Prüfe X: sollte von 0 zu >0 gehen
            x_values = [d.get('persons_detected', 0) for d in x_seq]
            if len(x_values) >= 2 and x_values[0] == 0 and max(x_values) > 0:
                clarity_score = 1.0
        
        # Gewichtete Kombination
        confidence = (
            0.35 * pattern_strength +
            0.30 * temporal_score +
            0.25 * avg_detection_conf +
            0.10 * clarity_score
        )
        
        return round(max(0.1, min(1.0, confidence)), 3)
    
    def _print_detection_summary(self, x_seq: List[Dict], y_seq: List[Dict]):
        """Zeigt Übersicht der Detections"""
        print(f"\n[MOVEMENT]   === Input X Detections (letzte 5) ===")
        for d in x_seq[-5:]:
            conf = d.get('avg_confidence')
            conf_str = f"{conf:.3f}" if isinstance(conf, (float, int)) and conf > 0 else "N/A"
            
            count = int(d.get('persons_detected', 0))
            ts = d.get('timestamp')
            time_str = ts.strftime('%H:%M:%S') if isinstance(ts, datetime) else "N/A"
            used = "USED" if d.get('used_in_pattern', False) else "FREE"
            
            print(f"[MOVEMENT]     ID={d['id']:4d} | {time_str} | {count:2d} Pers | Conf={conf_str:>5s} | {used}")
        
        print(f"\n[MOVEMENT]   === Input Y Detections (letzte 5) ===")
        for d in y_seq[-5:]:
            conf = d.get('avg_confidence')
            conf_str = f"{conf:.3f}" if isinstance(conf, (float, int)) and conf > 0 else "N/A"
            
            count = int(d.get('persons_detected', 0))
            ts = d.get('timestamp')
            time_str = ts.strftime('%H:%M:%S') if isinstance(ts, datetime) else "N/A"
            used = "USED" if d.get('used_in_pattern', False) else "FREE"
            
            print(f"[MOVEMENT]     ID={d['id']:4d} | {time_str} | {count:2d} Pers | Conf={conf_str:>5s} | {used}")


if __name__ == "__main__":
    from datetime import timedelta
    base_time = datetime.now()
    
    # Test: Exit-Muster (X→Y Sequenz)
    test_detections = [
        # X erscheint zuerst
        {'id': 1, 'timestamp': base_time, 'source': 'input_x', 'persons_detected': 0, 'avg_confidence': 0.0, 'used_in_pattern': False},
        {'id': 2, 'timestamp': base_time + timedelta(seconds=1), 'source': 'input_x', 'persons_detected': 1, 'avg_confidence': 0.9, 'used_in_pattern': False},
        {'id': 3, 'timestamp': base_time + timedelta(seconds=2), 'source': 'input_x', 'persons_detected': 1, 'avg_confidence': 0.9, 'used_in_pattern': False},
        {'id': 4, 'timestamp': base_time + timedelta(seconds=3), 'source': 'input_x', 'persons_detected': 0, 'avg_confidence': 0.0, 'used_in_pattern': False},
        
        # Dann Y
        {'id': 5, 'timestamp': base_time + timedelta(seconds=5), 'source': 'input_y', 'persons_detected': 0, 'avg_confidence': 0.0, 'used_in_pattern': False},
        {'id': 6, 'timestamp': base_time + timedelta(seconds=6), 'source': 'input_y', 'persons_detected': 1, 'avg_confidence': 0.88, 'used_in_pattern': False},
        {'id': 7, 'timestamp': base_time + timedelta(seconds=7), 'source': 'input_y', 'persons_detected': 1, 'avg_confidence': 0.88, 'used_in_pattern': False},
        {'id': 8, 'timestamp': base_time + timedelta(seconds=8), 'source': 'input_y', 'persons_detected': 0, 'avg_confidence': 0.0, 'used_in_pattern': False},
    ]
    
    detector = MovementDetector()
    moves = detector.detect_movements(test_detections)
    print("\n=== Erkannte Bewegungen ===")
    for m in moves:
        print(f"  {m['type'].upper()}: {m['person_count']} Person(en)")
        print(f"    Confidence: {m['confidence']:.2f}")
        print(f"    Pattern: {m['pattern']}")