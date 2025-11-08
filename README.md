# Live-Personenerkennung mit Bewegungsanalyse

## Überarbeitungen im Code

### 1. Entfernte Emojis und Sonderzeichen
- Alle Emojis in Log-Ausgaben durch Text ersetzt
- Umlaute durch ae/oe/ue/ss ersetzt
- Konsistente Verwendung von [TAG] Prefixen für Logs

### 2. Komprimierte Debug-Ausgaben

#### LiveProcessor
- Format: `[####] HH:MM:SS.mmm | source | ## Pers | Conf=X.XXX | X.XXXs [STATUS]`
- Beispiel: `[0042] 14:23:15.432 | input_x    |  3 Pers | Conf=0.823 | 0.145s [OK]`

#### MovementDetector
- Zeigt nur letzte 3 Detections pro Quelle
- Kompakte Entry/Exit-Ausgabe mit Muster
- Fokus auf erkannte Bewegungen
- Format: `[MOVEMENT] ENTRY erkannt: 2 Person(en), Conf=0.75`

#### TimeSeriesAnalyzer
- Reduzierte Ausgabe auf Wesentliches
- Klare Trennung durch Trennlinien
- Fokus auf Entry/Exit-Ereignisse
- Aktueller Raumzustand prominent dargestellt

### 3. Erweiterte Kommentare

Alle Klassen und Methoden haben jetzt Docstrings:
- Beschreibung der Funktionalität
- Args: Parameter-Dokumentation
- Returns: Rückgabewert-Dokumentation

Inline-Kommentare für komplexe Logik hinzugefügt.

### 4. Log-Präfixe

Konsistente Verwendung von Präfixen:
- `[DB]` - Datenbank-Operationen
- `[SYSTEM]` - System-Level Events
- `[MOVEMENT]` - Bewegungserkennung
- `[ROOM]` - Raumzustandsänderungen
- `[ANALYZER]` - Zeitreihenanalyse
- `[INFO]` - Informationen
- `[ERROR]` - Fehler
- `[OK]` - Erfolgreiche Operationen

## Kernfunktionalität

### Entry-Erkennung
1. Y-Detections steigen (z.B. 2 -> 3 Personen)
2. X-Detections zeigen Aktivität im Zeitfenster
3. Personenzahl = min(Y-Delta, X-Durchschnitt)
4. Log: `[MOVEMENT] Entry-Muster: Y+1 @ HH:MM:SS, X_avg=2.5 -> 1 Pers`

### Exit-Erkennung
1. Y-Detections fallen (z.B. 3 -> 2 Personen)
2. X-Detections zeigen Aktivität im Zeitfenster
3. Personenzahl = min(|Y-Delta|, X-Durchschnitt)
4. Log: `[MOVEMENT] Exit-Muster: Y-1 @ HH:MM:SS, X_avg=2.3 -> 1 Pers`

### Raumzustand-Update
```
[ROOM] ENTRY: 5 -> 6 (+1, Conf=0.75)
[ROOM] AKTUELLER ZUSTAND: 6 Personen im Raum
```

## Installation

```bash
# Python-Abhängigkeiten
pip install opencv-python ultralytics pg8000 numpy

# Skript ausführbar machen
chmod +x start_system.sh

# Datenbank-Konfiguration in start_system.sh anpassen
# oder in run_time_series_analysis.py (Zeilen 11-17)
```

## Verwendung

### System starten
```bash
./start_system.sh start
```

### Status prüfen
```bash
./start_system.sh status
```

### Logs verfolgen
```bash
./start_system.sh logs
```

### Raumzustand anzeigen
```bash
./start_system.sh room
```

### Statistiken anzeigen
```bash
./start_system.sh stats
```

### System stoppen
```bash
./start_system.sh stop
```

## Dateien-Übersicht

### Kern-Module
- `BaseDetector.py` - Abstrakte Basis für Detektoren
- `UltralyticsPersonDetector.py` - YOLO-basierte Personenerkennung
- `DataLoader.py` - Überwacht Ordner und lädt Bilder
- `DatabaseHandler.py` - PostgreSQL-Operationen
- `LiveProcessor.py` - Live-Verarbeitung von Bildern

### Bewegungsanalyse
- `MovementDetector.py` - Erkennt Entry/Exit aus Detections
- `RoomOccupancyManager.py` - Verwaltet Raumzustand
- `TimeSeriesAnalyzer.py` - Zeitreihenanalyse und Orchestrierung

### Startskripte
- `run_live_detection.py` - Startet Live-Detection
- `run_time_series_analysis.py` - Startet Bewegungsanalyse
- `start_system.sh` - Shell-Skript für Gesamtsystem

## Datenbank-Schema

### live_detections
- Rohdaten von beiden Kameras
- `processed` Flag für Verarbeitung
- JSONB für flexible Zusatzdaten

### movement_tracking
- Erkannte Bewegungen (entry/exit)
- Personenanzahl und Konfidenz
- Referenz zu beteiligten Detections

### room_state
- Zeitreihe des Raumzustands
- Grund der Änderung
- Referenz zu Bewegung (optional)

## Konfiguration

### Datenbank (start_system.sh)
```bash
DB_HOST="localhost"
DB_USER="aiuser"
DB_PASSWORD="DHBW1234!?"
DB_NAME="ai_detection"
DB_PORT=5432
```

### Eingabeordner
```bash
INPUT_X="input_x"
INPUT_Y="input_y"
```

### YOLO-Modell
```bash
YOLO_MODEL="yolov8n.pt"
CONFIDENCE_THRESHOLD=0.5
```

### Analyse-Parameter
- `transition_window=10.0` - Zeitfenster für Bewegungserkennung (Sekunden)
- `analysis_window=30.0` - Zeitfenster für unverarbeitete Detections (Sekunden)
- `min_confidence=0.3` - Mindest-Konfidenz für Logging
- `interval_seconds=30` - Intervall zwischen Analysen

## Beispiel-Output

### Live-Detection
```
[0001] 14:23:15.432 | input_x    |  5 Pers | Conf=0.823 | 0.145s [OK]
[0002] 14:23:15.987 | input_y    |  2 Pers | Conf=0.756 | 0.132s [OK]
[0003] 14:23:16.543 | input_x    |  6 Pers | Conf=0.845 | 0.138s [OK]
[0004] 14:23:17.098 | input_y    |  3 Pers | Conf=0.789 | 0.141s [OK]
```

### Bewegungsanalyse
```
[ANALYZER] Unverarbeitete Detections: 15
[ANALYZER] Input X: 8 | Input Y: 7
[ANALYZER] Letzte Detections:
  Input X (letzte 3):
    14:23:15 |  5 Pers | Conf=0.823
    14:23:16 |  6 Pers | Conf=0.845
    14:23:17 |  6 Pers | Conf=0.834
  Input Y (letzte 3):
    14:23:15 |  2 Pers | Conf=0.756
    14:23:17 |  3 Pers | Conf=0.789
    14:23:18 |  3 Pers | Conf=0.801

[MOVEMENT] Analyse gestartet: 15 Detections
[MOVEMENT] Zeitspanne: 3.5s
[MOVEMENT] Input X: 8 | Input Y: 7
[MOVEMENT] Entry-Muster: Y+1 @ 14:23:17, X_avg=5.7 -> 1 Pers
[MOVEMENT] ENTRY erkannt: 1 Person(en), Conf=0.75
[MOVEMENT] Gesamt: 1 Bewegung(en) erkannt

[ANALYZER] 1 Bewegung(en) verarbeiten:

  [1] ENTRY
      Personen: 1
      Konfidenz: 0.75
      Muster: Y+1, X_avg=5.7
      Gespeichert: ID=42

[ROOM] ENTRY: 5 -> 6 (+1, Conf=0.75)

[ANALYZER] 15 Detections als verarbeitet markiert

================================================================================
[ROOM] AKTUELLER ZUSTAND: 6 Personen im Raum
================================================================================
```

## Fehlerbehandlung

- Niedrige Konfidenz (<0.5): Bewegung wird ignoriert
- Implausible Änderungen: Werden abgelehnt
- Datenbankfehler: Werden geloggt, System läuft weiter
- Fehlende Bilder: Werden übersprungen

## Performance

- Live-Detection: ~0.15s pro Bild (YOLOv8n)
- Bewegungsanalyse: ~0.1s pro Zyklus
- Speicherbedarf: ~500MB (inkl. YOLO-Modell)

## Troubleshooting

### Keine Bewegungen erkannt
- Prüfe: Sind ausreichend Detections vorhanden? (min. 2 pro Quelle)
- Prüfe: Zeitfenster groß genug? (transition_window)
- Prüfe: Änderungen in Y-Detections vorhanden?

### Implausible Änderungen
- Max. Kapazität anpassen: `RoomOccupancyManager(max_capacity=100)`
- Max. Änderung pro Bewegung: derzeit 10 Personen

### Datenbankverbindung
```bash
# Test
./start_system.sh test-db

# Manuell
psql -U aiuser -h localhost -d ai_detection
```