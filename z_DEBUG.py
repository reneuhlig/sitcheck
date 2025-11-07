#!/usr/bin/env python3
"""
Debug-Skript für das Live-System
Prüft Status, zeigt Detections und Bewegungen
"""

import sys
from DatabaseHandler import DatabaseHandler
from datetime import datetime, timedelta


def main():
    db_config = {
        'host': 'localhost',
        'user': 'aiuser',
        'password': 'DHBW1234!?',
        'database': 'ai_detection',
        'port': 5432
    }
    
    db = DatabaseHandler(**db_config)
    
    if not db.connect():
        print("❌ Datenbankverbindung fehlgeschlagen")
        return
    
    cursor = db.connection.cursor()
    
    print("="*80)
    print("🔍 SYSTEM DEBUG")
    print("="*80)
    
    # 1. Prüfe letzte Detections
    print("\n📸 LETZTE 10 DETECTIONS:")
    print("-"*80)
    cursor.execute("""
        SELECT id, timestamp, source, persons_detected, avg_confidence, processed
        FROM live_detections
        ORDER BY timestamp DESC
        LIMIT 10
    """)
    
    for row in cursor.fetchall():
        conf = f"{row[4]:.3f}" if row[4] is not None else "N/A"
        status = "✓ verarbeitet" if row[5] else "⏳ unverarbeitet"
        print(f"ID {row[0]:4d} | {row[1]} | {row[2]:10s} | "
            f"{row[3]:2d} Personen | Conf: {conf} | {status}")
    # 2. Prüfe unverarbeitete Detections
    print("\n⏳ UNVERARBEITETE DETECTIONS:")
    print("-"*80)
    cursor.execute("""
        SELECT COUNT(*), source
        FROM live_detections
        WHERE processed = FALSE
        GROUP BY source
    """)
    
    unprocessed_total = 0
    for row in cursor.fetchall():
        print(f"  {row[1]}: {row[0]} unverarbeitet")
        unprocessed_total += row[0]
    
    if unprocessed_total == 0:
        print("  ✓ Alle Detections verarbeitet")
    else:
        print(f"\n  ⚠️  Gesamt: {unprocessed_total} unverarbeitete Detections!")
        print(f"     → Analyzer läuft möglicherweise nicht!")
    
    # 3. Prüfe erkannte Bewegungen
    print("\n🎯 LETZTE 10 BEWEGUNGEN:")
    print("-"*80)
    cursor.execute("""
        SELECT id, timestamp, movement_type, person_count, confidence_score, notes
        FROM movement_tracking
        ORDER BY timestamp DESC
        LIMIT 10
    """)
    
    movements = cursor.fetchall()
    if not movements:
        print("  ❌ KEINE BEWEGUNGEN GEFUNDEN!")
        print("     → Analyzer erkennt keine Muster oder läuft nicht")
    else:
        for row in movements:
            emoji = "🟢" if row[2] == 'entry' else "🔴" if row[2] == 'exit' else "❓"
            print(f"{emoji} ID {row[0]:4d} | {row[1]} | {row[2]:12s} | "
                  f"{row[3]:2d} Personen | Conf: {row[4]:.3f} | {row[5]}")
    
    # 4. Prüfe Raumzustand
    print("\n📊 LETZTE 5 RAUMZUSTÄNDE:")
    print("-"*80)
    cursor.execute("""
        SELECT id, timestamp, total_persons, change_reason, confidence, notes
        FROM room_state
        ORDER BY timestamp DESC
        LIMIT 5
    """)
    
    states = cursor.fetchall()
    if not states:
        print("  ❌ KEIN RAUMZUSTAND GEFUNDEN!")
        print("     → System wurde nicht initialisiert")
    else:
        for row in states:
            print(f"ID {row[0]:4d} | {row[1]} | {row[2]:3d} Personen | "
                  f"{row[3]:15s} | Conf: {row[4] if row[4] else 'N/A'}")
            if row[5]:
                print(f"         └─ {row[5]}")
    
    # 5. Zeitfenster-Analyse
    print("\n⏱️  AKTIVITÄT LETZTE 5 MINUTEN:")
    print("-"*80)
    
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE source = 'input_x') as x_count,
            COUNT(*) FILTER (WHERE source = 'input_y') as y_count,
            COUNT(*) FILTER (WHERE processed = TRUE) as processed,
            COUNT(*) FILTER (WHERE processed = FALSE) as unprocessed
        FROM live_detections
        WHERE timestamp > NOW() - INTERVAL '5 minutes'
    """)
    
    row = cursor.fetchone()
    print(f"  Detections gesamt: {row[0]}")
    print(f"  ├─ Input X: {row[1]}")
    print(f"  ├─ Input Y: {row[2]}")
    print(f"  ├─ Verarbeitet: {row[3]}")
    print(f"  └─ Unverarbeitet: {row[4]}")
    
    cursor.execute("""
        SELECT COUNT(*), SUM(person_count)
        FROM movement_tracking
        WHERE timestamp > NOW() - INTERVAL '5 minutes'
    """)
    
    row = cursor.fetchone()
    print(f"\n  Bewegungen erkannt: {row[0]}")
    if row[1]:
        print(f"  └─ Personen-Delta gesamt: {row[1]}")
    
    # 6. Detections-Rate
    print("\n📈 DETECTIONS PRO MINUTE (letzte 10 Min):")
    print("-"*80)
    cursor.execute("""
        SELECT 
            DATE_TRUNC('minute', timestamp) as minute,
            COUNT(*) as count,
            COUNT(*) FILTER (WHERE source = 'input_x') as x_count,
            COUNT(*) FILTER (WHERE source = 'input_y') as y_count
        FROM live_detections
        WHERE timestamp > NOW() - INTERVAL '10 minutes'
        GROUP BY minute
        ORDER BY minute DESC
        LIMIT 10
    """)
    
    for row in cursor.fetchall():
        print(f"  {row[0]} | Gesamt: {row[1]:3d} | X: {row[2]:3d} | Y: {row[3]:3d}")
    
    # 7. Empfehlungen
    print("\n💡 EMPFEHLUNGEN:")
    print("-"*80)
    
    if unprocessed_total > 50:
        print("  ⚠️  Zu viele unverarbeitete Detections!")
        print("     → Prüfe ob Analyzer läuft: ps aux | grep time_series")
        print("     → Starte Analyzer neu: python3 run_time_series_analysis.py")
    
    if not movements:
        print("  ⚠️  Keine Bewegungen erkannt!")
        print("     → Prüfe MovementDetector-Logik")
        print("     → Reduziere min_detections in TimeSeriesAnalyzer")
        print("     → Erhöhe analysis_window in TimeSeriesAnalyzer")
    
    if not states:
        print("  ⚠️  Kein Raumzustand!")
        print("     → Starte Analyzer: python3 run_time_series_analysis.py")
    
    if unprocessed_total == 0 and not movements:
        print("  ⚠️  Detections werden verarbeitet, aber keine Bewegungen erkannt!")
        print("     → Erhöhe transition_window in MovementDetector")
        print("     → Reduziere min_confidence in MovementDetector")
        print("     → Prüfe ob Muster überhaupt erkennbar sind (siehe Detections oben)")
    
    print("\n" + "="*80)
    
    cursor.close()
    db.close()


if __name__ == "__main__":
    main()