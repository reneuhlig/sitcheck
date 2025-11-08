#!/usr/bin/env python3
"""
Startskript für  Bewegungsanalyse
"""

import sys
from TimeSeriesAnalyzer import TimeSeriesAnalyzer


def main():
    """
    Startet die Bewegungsanalyse
    """
    # Datenbank-Konfiguration
    db_config = {
        'host': 'localhost',
        'user': 'aiuser',
        'password': 'DHBW1234!?',
        'database': 'ai_detection',
        'port': 5432
    }
    
    print("  Starte Bewegungsanalyse...")
    print(f"   Datenbank: {db_config['host']}:{db_config['port']}/{db_config['database']}")

    
    try:
        analyzer = TimeSeriesAnalyzer(db_config)
        
        # Starte kontinuierliche Analyse
        analyzer.start(interval_seconds=30, continuous=True)
        
    except KeyboardInterrupt:
        print("\n\nINFO: Programm beendet")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: Fehler: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()