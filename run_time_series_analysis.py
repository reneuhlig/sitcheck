#!/usr/bin/env python3
"""
Startskript fuer Bewegungsanalyse
"""

import sys
from TimeSeriesAnalyzer import TimeSeriesAnalyzer


def main():
    """Startet die Bewegungsanalyse"""
    # Datenbank-Konfiguration
    db_config = {
        'host': 'localhost',
        'user': 'aiuser',
        'password': 'DHBW1234!?',
        'database': 'ai_detection',
        'port': 5432
    }
    
    print("[INFO] Starte Bewegungsanalyse...")
    print(f"[INFO] Datenbank: {db_config['host']}:{db_config['port']}/{db_config['database']}")

    
    try:
        analyzer = TimeSeriesAnalyzer(db_config)
        
        # Starte kontinuierliche Analyse
        analyzer.start(interval_seconds=30, continuous=True)
        
    except KeyboardInterrupt:
        print("\n\n[INFO] Programm beendet")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Fehler: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()