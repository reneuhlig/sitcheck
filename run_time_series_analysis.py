#!/usr/bin/env python3
"""
Einfaches Startskript für die verbesserte Bewegungsanalyse
"""

import sys
from TimeSeriesAnalyzer import TimeSeriesAnalyzer


def main():
    """
    Startet die Bewegungsanalyse mit fest konfigurierten Werten
    """
    # Datenbank-Konfiguration (ANPASSEN!)
    db_config = {
        'host': 'localhost',
        'user': 'aiuser',
        'password': 'DHBW1234!?',
        'database': 'ai_detection',
        'port': 5432
    }
    
    print("🚀 Starte Bewegungsanalyse...")
    print(f"   Datenbank: {db_config['host']}:{db_config['port']}/{db_config['database']}")
    print()
    
    try:
        analyzer = TimeSeriesAnalyzer(db_config)
        
        # Starte kontinuierliche Analyse
        # interval_seconds: Wie oft nach neuen Detections schauen
        # continuous: True = läuft dauerhaft, False = nur einmal
        analyzer.start(interval_seconds=2, continuous=True)
        
    except KeyboardInterrupt:
        print("\n\n👋 Programm beendet")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()