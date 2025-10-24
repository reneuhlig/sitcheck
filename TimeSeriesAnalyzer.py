#!/usr/bin/env python3
"""
Zeitreihenanalyse für Personenzählungen aus zwei Quellen
Korreliert die Daten und schätzt die tatsächliche Personenanzahl
"""

import time
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import statistics
from decimal import Decimal
from DatabaseHandler import DatabaseHandler


class TimeSeriesAnalyzer:
    """Analysiert Zeitreihen von Personenzählungen"""
    
    def __init__(self, db_config: Dict[str, str]):
        """
        Initialisiert den Analyzer
        
        Args:
            db_config: Datenbank-Konfiguration
        """
        self.db = DatabaseHandler(**db_config)
        
        # Analyse-Parameter
        self.max_time_diff = 5.0  # Max. Zeitunterschied für Paare (Sekunden)
        self.confidence_threshold = 0.5  # Mindest-Konfidenz
        
    def analyze_and_store(self, interval_seconds: int = 10, continuous: bool = True):
        """
        Führt kontinuierliche Analyse durch und speichert Ergebnisse
        
        Args:
            interval_seconds: Intervall zwischen Analysen
            continuous: Kontinuierlich laufen oder nur einmal
        """
        if not self.db.connect():
            print("✗ Datenbankverbindung fehlgeschlagen")
            return
        
        print(f"\n{'='*80}")
        print(f"📊 ZEITREIHENANALYSE GESTARTET")
        print(f"{'='*80}")
        print(f"  Analyse-Intervall: {interval_seconds}s")
        print(f"  Max. Zeitdifferenz: {self.max_time_diff}s")
        print(f"  Konfidenz-Schwelle: {self.confidence_threshold}")
        print(f"{'='*80}\n")
        
        analysis_count = 0
        
        try:
            while True:
                analysis_count += 1
                print(f"\n[Analyse #{analysis_count}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("-" * 80)
                
                # Gepaarte Detections abrufen
                pairs = self.db.get_paired_detections(
                    max_time_diff_seconds=self.max_time_diff,
                    limit=100
                )
                
                if not pairs:
                    print("⚠ Keine passenden Paare gefunden")
                else:
                    print(f"✓ {len(pairs)} Paare gefunden")
                    
                    # Analysiere jedes Paar
                    results = []
                    for pair in pairs:
                        result = self._analyze_pair(pair)
                        if result:
                            results.append(result)
                    
                    # Speichere Ergebnisse
                    saved_count = 0
                    for result in results:
                        if self.db.insert_correlated_result(**result):
                            saved_count += 1
                    
                    print(f"✓ {saved_count}/{len(results)} Ergebnisse gespeichert")
                    
                    # Statistik ausgeben
                    if results:
                        self._print_statistics(results)
                
                if not continuous:
                    break
                
                print(f"\n⏳ Warte {interval_seconds}s bis zur nächsten Analyse...")
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\n\n❌ Analyse durch Benutzer abgebrochen")
        finally:
            self.db.close()
    
    def _analyze_pair(self, pair: Dict) -> Dict:
        """
        Analysiert ein Paar von Detections
        
        Returns:
            Dictionary mit Analyse-Ergebnis oder None
        """
        x_persons = pair['x_persons']
        y_persons = pair['y_persons']
        x_conf = float(pair['x_confidence']) if pair['x_confidence'] is not None else 0.0
        y_conf = float(pair['y_confidence']) if pair['y_confidence'] is not None else 0.0
        time_diff = float(abs(pair['time_diff']))
        
        # Qualitätsprüfung
        if x_conf < self.confidence_threshold or y_conf < self.confidence_threshold:
            return None
        
        # Zeitdifferenz-Strafe (je größer die Differenz, desto weniger vertrauenswürdig)
        time_penalty = max(0.0, 1.0 - (time_diff / self.max_time_diff))
        
        # Gewichtete Durchschnittsberechnung basierend auf Konfidenz
        total_weight = x_conf + y_conf
        weighted_persons = (x_persons * x_conf + y_persons * y_conf) / total_weight
        
        # Tatsächliche Personenanzahl schätzen (verschiedene Strategien)
        estimated_persons = self._estimate_actual_persons(
            x_persons, y_persons, x_conf, y_conf
        )
        
        # Konfidenz des Ergebnisses
        result_confidence = (x_conf + y_conf) / 2 * time_penalty
        
        # Analyse-Daten für Nachvollziehbarkeit
        analysis_data = {
            'method': 'weighted_average',
            'weighted_persons': round(weighted_persons, 2),
            'time_penalty': round(time_penalty, 3),
            'x_confidence': round(x_conf, 3),
            'y_confidence': round(y_conf, 3),
            'difference': abs(x_persons - y_persons),
            'agreement': x_persons == y_persons
        }
        
        return {
            'source_x_id': pair['x_id'],
            'source_y_id': pair['y_id'],
            'persons_x': x_persons,
            'persons_y': y_persons,
            'estimated_actual': estimated_persons,
            'confidence': result_confidence,
            'time_diff': time_diff,
            'analysis_data': analysis_data
        }
    
    def _estimate_actual_persons(self, x_persons: int, y_persons: int,
                                 x_conf: float, y_conf: float) -> int:
        """
        Schätzt die tatsächliche Personenanzahl
        
        Strategien:
        1. Bei Übereinstimmung: Wert übernehmen
        2. Bei Abweichung: Höhere Konfidenz gewinnt
        3. Bei gleicher Konfidenz: Durchschnitt (gerundet)
        4. Bei großer Abweichung: Maximum nehmen (konservativ)
        """
        # Strategie 1: Perfekte Übereinstimmung
        if x_persons == y_persons:
            return x_persons
        
        # Strategie 2: Unterschied > 2 Personen -> nehme Maximum (konservativ)
        if abs(x_persons - y_persons) > 2:
            return max(x_persons, y_persons)
        
        # Strategie 3: Konfidenz entscheidet
        conf_diff = abs(x_conf - y_conf)
        if conf_diff > 0.1:  # Signifikanter Unterschied
            return x_persons if x_conf > y_conf else y_persons
        
        # Strategie 4: Gewichteter Durchschnitt
        weighted = (x_persons * x_conf + y_persons * y_conf) / (x_conf + y_conf)
        return round(weighted)
    
    def _print_statistics(self, results: List[Dict]):
        """Gibt Statistiken über die Analyseergebnisse aus"""
        if not results:
            return
        
        # Extrahiere Werte
        estimated = [r['estimated_actual'] for r in results]
        confidences = [r['confidence'] for r in results]
        differences = [r['analysis_data']['difference'] for r in results]
        agreements = sum(1 for r in results if r['analysis_data']['agreement'])
        
        print("\n📈 STATISTIK:")
        print(f"  Übereinstimmungen: {agreements}/{len(results)} ({agreements/len(results)*100:.1f}%)")
        print(f"  Durchschn. Differenz: {statistics.mean(differences):.2f} Personen")
        print(f"  Max. Differenz: {max(differences)} Personen")
        print(f"  Durchschn. geschätzte Personen: {statistics.mean(estimated):.2f}")
        print(f"  Min/Max geschätzt: {min(estimated)}/{max(estimated)} Personen")
        print(f"  Durchschn. Konfidenz: {statistics.mean(confidences):.3f}")
    
    def get_recent_summary(self, hours: int = 1) -> Dict:
        """
        Gibt eine Zusammenfassung der letzten Stunden zurück
        
        Args:
            hours: Zeitraum in Stunden
            
        Returns:
            Dictionary mit Zusammenfassung
        """
        if not self.db.connection:
            self.db.connect()
        
        cursor = self.db.connection.cursor()
        query = """
        SELECT 
            COUNT(*) as total_correlations,
            AVG(estimated_actual_persons) as avg_persons,
            MIN(estimated_actual_persons) as min_persons,
            MAX(estimated_actual_persons) as max_persons,
            AVG(confidence_score) as avg_confidence,
            AVG(time_diff_seconds) as avg_time_diff
        FROM correlated_persons
        WHERE timestamp >= NOW() - INTERVAL '%s hours'
        """
        
        try:
            cursor.execute(query, (hours,))
            row = cursor.fetchone()
            
            if row and row[0] > 0:
                return {
                    'total_correlations': row[0],
                    'avg_persons': round(float(row[1]), 2) if row[1] else 0,
                    'min_persons': row[2] or 0,
                    'max_persons': row[3] or 0,
                    'avg_confidence': round(float(row[4]), 3) if row[4] else 0,
                    'avg_time_diff': round(float(row[5]), 3) if row[5] else 0
                }
            else:
                return {'message': 'Keine Daten im angegebenen Zeitraum'}
                
        except Exception as e:
            print(f"✗ Fehler beim Abrufen der Zusammenfassung: {e}")
            return {}
        finally:
            cursor.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Zeitreihenanalyse für Personenzählungen')
    parser.add_argument('--db-host', default='localhost', help='PostgreSQL Host')
    parser.add_argument('--db-user', required=True, help='PostgreSQL Benutzername')
    parser.add_argument('--db-password', required=True, help='PostgreSQL Passwort')
    parser.add_argument('--db-name', required=True, help='PostgreSQL Datenbankname')
    parser.add_argument('--db-port', type=int, default=5432, help='PostgreSQL Port')
    parser.add_argument('--interval', type=int, default=10, help='Analyse-Intervall (Sekunden)')
    parser.add_argument('--once', action='store_true', help='Nur eine Analyse durchführen')
    parser.add_argument('--summary', type=int, help='Zeige Zusammenfassung der letzten N Stunden')
    
    args = parser.parse_args()
    
    db_config = {
        'host': args.db_host,
        'user': args.db_user,
        'password': args.db_password,
        'database': args.db_name,
        'port': args.db_port
    }
    
    analyzer = TimeSeriesAnalyzer(db_config)
    
    if args.summary:
        # Zeige nur Zusammenfassung
        if analyzer.db.connect():
            summary = analyzer.get_recent_summary(hours=args.summary)
            print(f"\n📊 ZUSAMMENFASSUNG (letzte {args.summary} Stunde(n)):")
            print("="*60)
            for key, value in summary.items():
                print(f"  {key}: {value}")
            analyzer.db.close()
    else:
        # Starte Analyse
        analyzer.analyze_and_store(
            interval_seconds=args.interval,
            continuous=not args.once
        )