System-Dokumentation: Live-Personenzählung mit Bewegungserkennung
Inhaltsverzeichnis

Systemübersicht
Architektur
Komponenten
Datenfluss
Installation & Konfiguration
Verwendung
Datenbank-Schema
Algorithmen
Fehlerbehebung


Systemübersicht
Zweck
Das System zählt Personen in einem Raum durch Analyse von zwei Kamerastreams an Ein-/Ausgang. Es erkennt automatisch Entry- und Exit-Bewegungen und verwaltet die aktuelle Raumbelegung.
Hauptfunktionen

Live-Personenerkennung mit YOLOv8
Bewegungsmustererkennung (Entry/Exit)
Raumzustandsverwaltung mit Plausibilitätsprüfungen
Zeitreihenanalyse für robuste Mustererkennung
PostgreSQL-Persistierung aller Detections und Bewegungen

Technologie-Stack

Python 3.x
Ultralytics YOLOv8 (Personenerkennung)
OpenCV (Bildverarbeitung)
PostgreSQL (Datenspeicherung)
pg8000 (Datenbank-Treiber)
PowerShell (Kamera-Capture, Windows)


Architektur
Systemkomponenten-Diagramm
┌─────────────────────────────────────────────────────────────┐
│                    KAMERA-CAPTURE-SYSTEM                      │
│                     (Camera.ps1)                              │
└────────────────┬────────────────────────┬───────────────────┘
                 │                        │
                 ▼                        ▼
         ┌──────────────┐        ┌──────────────┐
         │  input_x/    │        │  input_y/    │
         │  (Kamera X)  │        │  (Kamera Y)  │
         └──────┬───────┘        └──────┬───────┘
                │                        │
                └────────────┬───────────┘
                             ▼
                ┌─────────────────────────┐
                │    DataLoader           │
                │  (LiveImageLoader)      │
                └────────────┬────────────┘
                             ▼
                ┌─────────────────────────┐
                │   LiveProcessor         │
                │  (Personenerkennung)    │
                └────────────┬────────────┘
                             ▼
                ┌─────────────────────────┐
                │   UltralyticsDetector   │
                │      (YOLOv8)           │
                └────────────┬────────────┘
                             ▼
                ┌─────────────────────────┐
                │   PostgreSQL Database   │
                │  (live_detections)      │
                └────────────┬────────────┘
                             ▼
                ┌─────────────────────────┐
                │  TimeSeriesAnalyzer     │
                │  (Bewegungsanalyse)     │
                └────────────┬────────────┘
                             ▼
                ┌─────────────────────────┐
                │   MovementDetector      │
                │  (Mustererkennung)      │
                └────────────┬────────────┘
                             ▼
                ┌─────────────────────────┐
                │ RoomOccupancyManager    │
                │  (Raumzustand)          │
                └─────────────────────────┘

Komponenten
1. Kamera-Capture (Camera.ps1)
Zweck: Paralleles Erfassen von Bildern aus zwei Kameras und Upload zum Server
Hauptfunktionen:

Parallele Kamera-Captures mit Python/OpenCV
Batch-Upload via pscp.exe (PuTTY)
Queue-Management für Upload-Optimierung
Fehlerbehandlung und Statistiken

Konfiguration:
powershell$Config = @{
    ServerIP    = "192.168.194.65"
    CameraX     = 2                # Kamera-Index
    CameraY     = 3
    Interval    = 0.05             # Capture-Intervall
    BatchSize   = 1                # Upload-Batch-Größe
    Quality     = 80               # JPEG-Qualität
}
Wichtige Funktionen:

Invoke-CameraCapture: Einzelnes Bild erfassen
Send-FileBatch: Batch-Upload via SSH
Start-ParallelCapture: Gleichzeitiger Capture beider Kameras


2. DataLoader (DataLoader.py)
Zweck: Überwachung der Eingabeordner und Bereitstellung neuer Bilder
Klasse: LiveImageLoader
Hauptmethoden:
pythondef watch(self) -> Generator[tuple[str, any, Path], None, None]:
    """
    Überwacht Ordner und gibt Bilder zurück
    
    Yields:
        (source_name, image, file_path)
    """
pythondef confirm_processed(self, file_path: Path):
    """Löscht Datei nach erfolgreicher Verarbeitung"""
Features:

Automatische Erkennung neuer Bilddateien
Stabilitätsprüfung (verhindert Verarbeitung unvollständiger Uploads)
Unterstützung mehrerer Bildformate (.jpg, .png, .bmp, etc.)
Explizites Löschen nach Verarbeitung


3. LiveProcessor (LiveProcessor.py)
Zweck: Orchestrierung der Live-Bildverarbeitung
Klasse: LiveProcessor
Hauptmethoden:
pythondef start(self):
    """Startet Live-Überwachung und Verarbeitung"""
    
def _process_image(self, source: str, img, count: int) -> bool:
    """Verarbeitet einzelnes Bild mit Detektor"""
```

**Workflow**:
1. Bild von `DataLoader` empfangen
2. YOLOv8-Detection durchführen
3. Ergebnis in Datenbank speichern
4. Datei löschen
5. Kompakte Log-Ausgabe

**Log-Format**:
```
[0042] 14:23:15.742 | input_x    |  2 Pers | Conf: 0.876 | 0.123s [OK]

4. UltralyticsPersonDetector (UltralyticsPersonDetector.py)
Zweck: YOLOv8-basierte Personenerkennung
Klasse: UltralyticsPersonDetector
Hauptmethoden:
pythondef detect(self, image) -> Dict[str, Any]:
    """
    Erkennt Personen in einem Bild
    
    Returns:
        {
            'persons_detected': int,
            'persons': List[Dict],  # Bounding Boxes
            'avg_confidence': float,
            'max_confidence': float,
            'min_confidence': float
        }
    """
Konfiguration:

model_path: Pfad zum YOLOv8-Modell (z.B. "yolov8n.pt")
confidence_threshold: Mindest-Konfidenz (Standard: 0.5)
person_class_id: COCO-Klasse für Personen (0)


5. DatabaseHandler (DatabaseHandler.py)
Zweck: Verwaltung aller Datenbankoperationen
Klasse: DatabaseHandler
Wichtige Methoden:
pythondef insert_detection(self, source: str, persons_detected: int, ...):
    """Fügt neue Detection ein"""

def get_detections_for_analysis(self, analysis_window: float, 
                                 lookback_window: float):
    """Holt Detections für Bewegungsanalyse"""

def insert_movement(self, movement_type: str, person_count: int, ...):
    """Speichert erkannte Bewegung"""

def insert_room_state(self, total_persons: int, change_reason: str, ...):
    """Aktualisiert Raumzustand"""
```

**Tabellen**:
- `live_detections`: Rohdaten aller Detections
- `movement_tracking`: Erkannte Bewegungsmuster
- `room_state`: Historische Raumzustände
- `correlated_persons`: Legacy-Korrelationen (deprecated)

---

### 6. MovementDetector (`MovementDetector.py`)

**Zweck**: Sequenz-basierte Erkennung von Entry/Exit-Mustern

**Klasse**: `MovementDetector`

**Algorithmus-Prinzip**:

**Entry-Muster** (Person betritt Raum):
```
Y→X Sequenz:
1. Person erscheint ZUERST in Kamera Y (außen)
2. DANN wird Aktivität in Kamera X (innen) detektiert
3. Zeitfenster: ≤ 10 Sekunden
```

**Exit-Muster** (Person verlässt Raum):
```
X→Y Sequenz:
1. Person erscheint ZUERST in Kamera X (innen)
2. DANN wird Aktivität in Kamera Y (außen) detektiert
3. Zeitfenster: ≤ 10 Sekunden
Hauptmethoden:
pythondef detect_movements(self, detections: List[Dict]) -> List[Dict]:
    """
    Analysiert Detections und findet Bewegungsmuster
    
    Returns:
        List[{
            'type': 'entry' | 'exit',
            'person_count': int,
            'confidence': float,
            'time_diff': float,
            'pattern': str,
            'sequence': {'x_ids': [], 'y_ids': []}
        }]
    """
Pattern Confidence Berechnung:
pythonconfidence = (
    0.35 * pattern_strength +      # Stärke der Änderung
    0.30 * temporal_coherence +    # Zeitliche Konsistenz
    0.25 * detection_quality +     # Durchschnittliche Confidence
    0.10 * sequence_clarity        # Klarheit der Sequenz
)
Konfiguration:

transition_window: Max. Zeit zwischen X/Y-Events (10s)
min_pattern_confidence: Mindest-Konfidenz für Pattern (0.3)
max_patterns_per_type: Max. Entry/Exit pro Zyklus (3)


7. RoomOccupancyManager (RoomOccupancyManager.py)
Zweck: Verwaltung der aktuellen Raumbelegung mit Plausibilitätsprüfungen
Klasse: RoomOccupancyManager
Hauptmethoden:
pythondef initialize(self):
    """Lädt aktuellen Zustand aus DB oder setzt auf 0"""

def process_movement(self, movement: Dict, movement_id: int) -> bool:
    """
    Verarbeitet erkannte Bewegung
    
    Ablauf:
    1. Confidence-Prüfung (≥ 0.3)
    2. Neue Personenanzahl berechnen
    3. Plausibilitätsprüfung
    4. Datenbank-Update
    5. Erfolgs-/Fehler-Logging
    """
Plausibilitätsprüfungen:

Keine negativen Werte
Nicht über Maximalkapazität
Dynamische Änderungs-Grenzen basierend auf Confidence:

Confidence ≥ 0.7: Max. ±10 Personen
Confidence ≥ 0.5: Max. ±5 Personen
Confidence ≥ 0.3: Max. ±3 Personen



Manuelle Korrekturen:
pythondef manual_correction(self, correct_count: int, reason: str):
    """Manuelle Korrektur bei Fehlern"""

def reset(self):
    """Reset auf 0 (z.B. Ende des Tages)"""

8. TimeSeriesAnalyzer (TimeSeriesAnalyzer.py)
Zweck: Kontinuierliche Analyse der Detections mit Bewegungserkennung
Klasse: TimeSeriesAnalyzer
Hauptmethoden:
pythondef start(self, interval_seconds: int = 30, continuous: bool = True):
    """
    Startet kontinuierliche Analyse
    
    Ablauf pro Zyklus:
    1. Detections aus DB holen (mit Lookback)
    2. Bewegungsmuster erkennen
    3. Raumzustand aktualisieren
    4. Detections als verarbeitet markieren
    """
Analyse-Parameter:

analysis_window: Zeitfenster für neue Detections (30s)
lookback_window: Zusätzliches Fenster für alte Detections (60s)
min_detections: Mindest-Anzahl Detections (1)

Zyklusübergreifende Mustererkennung:
Das System kann Muster erkennen, die sich über mehrere Analysezyklen erstrecken:
python# Beispiel: Entry-Muster
Zyklus N:   Y=0, Y=1 (außen)
Zyklus N+1: X=1 (innen)
→ Entry erkannt!
```

**Log-Ausgabe**:
```
[ANALYZER] Detections: 45 total
[ANALYZER]   Neu: 15
[ANALYZER]   Lookback (wiederverwendbar): 12
[ANALYZER]   Bereits in Muster: 18
```

---

## Datenfluss

### Vollständiger Ablauf
```
1. KAMERA-CAPTURE (Camera.ps1)
   ↓
   Bilder → input_x/, input_y/

2. LIVE-DETECTION (LiveProcessor)
   ↓
   DataLoader überwacht Ordner
   ↓
   YOLOv8 erkennt Personen
   ↓
   Speichern in live_detections
   ↓
   Datei löschen

3. ZEITREIHEN-ANALYSE (TimeSeriesAnalyzer)
   ↓
   Hole Detections (30s + 60s Lookback)
   ↓
   MovementDetector analysiert Sequenzen
   ↓
   Entry/Exit-Muster erkannt?
   ↓
   Ja: RoomOccupancyManager aktualisiert
   ↓
   Speichern in movement_tracking + room_state
   ↓
   Markiere Detections als verarbeitet

4. AUSGABE
   ↓
   Logs, Datenbank, Statistiken

Installation & Konfiguration
Systemvoraussetzungen
bash# Python 3.x
python3 --version

# PostgreSQL 12+
psql --version

# PuTTY Tools (Windows, für Camera.ps1)
pscp.exe --version
Python-Abhängigkeiten
bashpip install opencv-python
pip install ultralytics
pip install pg8000
pip install numpy
Datenbank-Setup
sql-- Datenbank erstellen
CREATE DATABASE ai_detection;

-- Benutzer erstellen
CREATE USER aiuser WITH PASSWORD 'DHBW1234!?';
GRANT ALL PRIVILEGES ON DATABASE ai_detection TO aiuser;
Konfiguration
1. Datenbank-Konfiguration (run_live_detection.py, run_time_series_analysis.py):
pythondb_config = {
    'host': 'localhost',
    'user': 'aiuser',
    'password': 'DHBW1234!?',
    'database': 'ai_detection',
    'port': 5432
}
2. Kamera-Konfiguration (Camera.ps1):
powershell$Config = @{
    ServerIP    = "192.168.194.65"
    ServerUser  = "kiadmin"
    ServerPass  = "DHBW1234!?"
    RemotePathX = "/sitcheck/input_x"
    RemotePathY = "/sitcheck/input_y"
    CameraX     = 2
    CameraY     = 3
}
3. YOLO-Modell:
bash# Modell herunterladen (automatisch beim ersten Start)
# Oder manuell:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

Verwendung
Start des kompletten Systems (Linux)
bash# System starten
./start_system.sh start

# Status prüfen
./start_system.sh status

# Logs verfolgen
./start_system.sh logs

# Raumzustand anzeigen
./start_system.sh room

# Statistiken anzeigen
./start_system.sh stats

# System stoppen
./start_system.sh stop
Manuelle Einzelkomponenten
Live-Detection starten:
bashpython3 run_live_detection.py \
    --db-user aiuser \
    --db-password "DHBW1234!?" \
    --db-name ai_detection \
    --input-x input_x \
    --input-y input_y \
    --yolo-model yolov8n.pt \
    --confidence-threshold 0.5 \
    --poll-interval 0.5 \
    --verbose
Bewegungsanalyse starten:
bashpython3 run_time_series_analysis.py
Kamera-Capture starten (Windows):
powershell.\Camera.ps1
Datenbank-Abfragen
Aktueller Raumzustand:
sqlSELECT total_persons, timestamp, change_reason, confidence
FROM room_state
ORDER BY timestamp DESC
LIMIT 1;
Letzte Bewegungen:
sqlSELECT timestamp, movement_type, person_count, confidence_score
FROM movement_tracking
ORDER BY timestamp DESC
LIMIT 10;
Detections der letzten 5 Minuten:
sqlSELECT source, COUNT(*), AVG(persons_detected), AVG(avg_confidence)
FROM live_detections
WHERE timestamp > NOW() - INTERVAL '5 minutes'
GROUP BY source;
Unverarbeitete Detections:
sqlSELECT COUNT(*) 
FROM live_detections 
WHERE processed = FALSE;

Datenbank-Schema
Tabelle: live_detections
Speichert alle Roh-Detections beider Kameras.
sqlCREATE TABLE live_detections (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source VARCHAR(50) NOT NULL,              -- 'input_x' oder 'input_y'
    persons_detected INTEGER NOT NULL,
    avg_confidence REAL,
    max_confidence REAL,
    min_confidence REAL,
    detection_data JSONB,                     -- Bounding Boxes, etc.
    processed BOOLEAN DEFAULT FALSE,          -- Bereits analysiert?
    used_in_pattern BOOLEAN DEFAULT FALSE,    -- In Bewegungsmuster verwendet?
    pattern_id INTEGER                        -- Referenz zu movement_tracking
);

CREATE INDEX idx_detections_timestamp ON live_detections (timestamp);
CREATE INDEX idx_detections_source ON live_detections (source);
CREATE INDEX idx_detections_processed ON live_detections (processed);
Beispiel-Eintrag:
json{
    "id": 1234,
    "timestamp": "2025-01-15 14:23:45",
    "source": "input_x",
    "persons_detected": 2,
    "avg_confidence": 0.876,
    "detection_data": {
        "persons": [
            {"bbox": [120, 80, 240, 360], "confidence": 0.89},
            {"bbox": [300, 90, 420, 380], "confidence": 0.86}
        ]
    },
    "processed": true,
    "used_in_pattern": true,
    "pattern_id": 456
}

Tabelle: movement_tracking
Speichert erkannte Bewegungsmuster.
sqlCREATE TABLE movement_tracking (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    movement_type VARCHAR(20) NOT NULL,       -- 'entry', 'exit'
    person_count INTEGER NOT NULL,
    confidence_score REAL,
    detection_sequence JSONB,                 -- Verwendete Detection-IDs
    notes TEXT
);

CREATE INDEX idx_movement_timestamp ON movement_tracking (timestamp);
Beispiel-Eintrag:
json{
    "id": 456,
    "timestamp": "2025-01-15 14:23:50",
    "movement_type": "entry",
    "person_count": 1,
    "confidence_score": 0.742,
    "detection_sequence": {
        "x_ids": [1235, 1236],
        "y_ids": [1230, 1231, 1232]
    },
    "notes": "Pattern: Y→X: Y increase +1 @ 14:23:45, then X avg 1.0"
}

Tabelle: room_state
Historische Raum-Belegungszustände.
sqlCREATE TABLE room_state (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_persons INTEGER NOT NULL,
    change_reason VARCHAR(50),                -- 'entry', 'exit', 'initialization', etc.
    movement_tracking_id INTEGER REFERENCES movement_tracking(id),
    confidence REAL,
    notes TEXT
);

CREATE INDEX idx_room_state_timestamp ON room_state (timestamp);
Beispiel-Einträge:
json[
    {
        "id": 1,
        "timestamp": "2025-01-15 14:00:00",
        "total_persons": 0,
        "change_reason": "initialization",
        "confidence": 1.0,
        "notes": "Initialisierung des Systems"
    },
    {
        "id": 2,
        "timestamp": "2025-01-15 14:23:50",
        "total_persons": 1,
        "change_reason": "entry",
        "movement_tracking_id": 456,
        "confidence": 0.742,
        "notes": "Delta: 1, Pattern: Y→X..."
    }
]

Algorithmen
1. Bewegungserkennung (MovementDetector)
Entry-Erkennung
Algorithmus:

Finde alle Y-Anstiege (0→N oder N→M)
Sortiere nach Stärke (größtes Delta zuerst)
Für jeden Y-Anstieg:

Suche X-Aktivität NACH Y-Anstieg (≤ 10s)
Prüfe ob X Personen detektiert
Berechne Pattern Confidence
Wenn Confidence ≥ 0.3: Entry erkannt



Code-Beispiel:
pythondef _detect_entry_pattern(self, x_seq, y_seq):
    y_increases = self._find_all_increases(y_seq)
    
    for y_time, y_delta, y_idx in y_increases:
        # Suche X NACH Y
        x_after_y = [d for d in x_seq 
                     if d['timestamp'] > y_time 
                     and (d['timestamp'] - y_time).total_seconds() < 10]
        
        if x_after_y and sum(d['persons_detected'] for d in x_after_y) > 0:
            confidence = self._calculate_pattern_confidence(...)
            if confidence >= 0.3:
                return {'type': 'entry', ...}
    return None
```

---

#### Exit-Erkennung

**Algorithmus**:
1. Finde alle X-Anstiege (0→N oder N→M)
2. Sortiere nach Stärke
3. Für jeden X-Anstieg:
   - Suche Y-Aktivität NACH X-Anstieg (≤ 10s)
   - Prüfe ob Y Personen detektiert
   - Berechne Pattern Confidence
   - Wenn Confidence ≥ 0.3: Exit erkannt

---

#### Pattern Confidence

**Formel**:
```
Confidence = 0.35 * Pattern_Strength 
           + 0.30 * Temporal_Coherence 
           + 0.25 * Detection_Quality 
           + 0.10 * Sequence_Clarity
```

**Komponenten**:

1. **Pattern Strength** (35%):
```
   1 Person:  0.4
   2 Personen: 0.7
   3+ Personen: 1.0
```

2. **Temporal Coherence** (30%):
```
   Zeit ≤ 10s:  1.0
   Zeit ≤ 20s:  1.0 - (Zeit - 10) / 10 * 0.5
   Zeit > 20s:  max(0.3, 0.5 - (Zeit - 20) / 30 * 0.2)
```

3. **Detection Quality** (25%):
```
   Durchschnittliche Detection-Confidence aller beteiligten Detections
```

4. **Sequence Clarity** (10%):
```
   Base: 0.8
   Bonus: 1.0 bei klarem 0→1 Übergang
```

---

### 2. Raumzustands-Management

**Ablauf**:
```
1. Movement erkannt (z.B. Entry, +1 Person, Conf=0.742)
   ↓
2. Confidence-Check (≥ 0.3)?
   ↓ Ja
3. Neue Personenanzahl berechnen
   Aktuell: 5 → Neu: 6
   ↓
4. Plausibilitäts-Checks:
   - Nicht negativ? ✓
   - Unter Max-Kapazität? ✓
   - Änderung ≤ Max-Delta für Confidence? ✓
   ↓ Alle OK
5. Datenbank-Update
   ↓
6. Interner Zustand aktualisiert
   ↓
7. Log-Ausgabe
```

---

### 3. Zyklusübergreifende Mustererkennung

**Problem**: Bewegungen können sich über mehrere Analysezyklen erstrecken.

**Lösung**: Lookback-Window
```
Zyklus N (T=0-30s):
  Y: 0, 0, 1, 1 (Person außen)
  → Keine X-Aktivität in diesem Zyklus
  → Detections als "processed=true" markiert, ABER "used_in_pattern=false"

Zyklus N+1 (T=30-60s):
  Y: 1, 0 (Person weg)
  X: 0, 1, 1, 0 (Person innen!)
  
  → Lookback: Hole auch alte Y-Detections (processed=true, used=false)
  → Y→X Sequenz erkannt!
  → Entry +1 Person
  → Alle verwendeten Detections: used_in_pattern=true
Implementierung:
pythondef get_detections_for_analysis(self, analysis_window=30, lookback_window=60):
    query = """
    SELECT * FROM live_detections
    WHERE (
        (processed = FALSE AND timestamp >= NOW() - %s)
        OR
        (processed = TRUE AND used_in_pattern = FALSE 
         AND timestamp >= NOW() - %s)
    )
    """
    cursor.execute(query, (analysis_window, lookback_window))

Fehlerbehebung
Häufige Probleme
1. Keine Detections in Datenbank
Symptome:

live_detections-Tabelle leer
Keine Logs von LiveProcessor

Diagnose:
bash# Prüfe ob LiveProcessor läuft
ps aux | grep run_live_detection

# Prüfe Logs
tail -f logs/detection.log

# Prüfe Ordner
ls -la input_x/ input_y/
Lösungen:

Kamera-Capture läuft nicht → Camera.ps1 starten
Falsche Ordner-Pfade → Konfiguration prüfen
YOLO-Modell fehlt → yolov8n.pt herunterladen
Datenbank-Verbindung → Connection-String prüfen


2. Keine Bewegungen erkannt
Symptome:

Detections vorhanden, aber movement_tracking leer
Raumzustand bleibt bei 0

Diagnose:
sql-- Prüfe unverarbeitete Detections
SELECT source, COUNT(*) 
FROM live_detections 
WHERE processed = FALSE 
GROUP BY source;

-- Prüfe Detection-Qualität
SELECT AVG(avg_confidence), MIN(avg_confidence) 
FROM live_detections 
WHERE timestamp > NOW() - INTERVAL '5 minutes';
Lösungen:

Zu wenig Detections → Mehr Zeit warten
Keine X-Y-Paare → Beide Kameras prüfen
Confidence zu niedrig → confidence_threshold senken
Pattern zu schwach → min_pattern_confidence senken (Vorsicht!)


3. Falsche Personenanzahl
Symptome:

Raumzustand weicht stark von Realität ab
Plötzliche große Sprünge

Diagnose:
sql-- Prüfe letzte Bewegungen
SELECT timestamp, movement_type, person_count, confidence_score 
FROM movement_tracking 
ORDER BY timestamp DESC 
LIMIT 20;

-- Prüfe Raumzustand-Historie
SELECT timestamp, total_persons, change_reason, confidence 
FROM room_state 
ORDER BY timestamp DESC 
LIMIT 20;
Lösungen:

Sofortige Korrektur: Manueller Reset

python  from RoomOccupancyManager import RoomOccupancyManager
  manager = RoomOccupancyManager(db, max_capacity=100)
  manager.initialize()
  manager.manual_correction(correct_count=5, reason="manuelle_korrektur")

Präventiv: Plausibilitäts-Grenzen anpassen
Dauerhaft: Kamera-Positionen optimieren


4. Hohe Fehlerquote
Symptome:

Viele "Implausible Änderung"-Meldungen
Confidence oft < 0.3

Diagnose:
python# Analyse der Pattern-Confidence
import pg8000
conn = pg8000.connect(...)
cursor = conn.cursor()
cursor.execute("""
    SELECT AVG(confidence_score), MIN(confidence_score), MAX(confidence_score)
    FROM movement_tracking
    WHERE timestamp > NOW() - INTERVAL '1 hour'
""")
print(cursor.fetchone())
Lösungen:

Kamera-Setup:

Beleuchtung verbessern
Kamera-Positionen anpassen
Höhere Auflösung


Detection-Parameter:

confidence_threshold erhöhen → Weniger False Positives
Größeres YOLO-Modell (yolov8m.pt, yolov8l.pt)


Pattern-Parameter:

transition_window anpassen (kürzer für schnelle Durchgänge)
min_pattern_confidence erhöhen (aber Vorsicht: weniger Detections)




5. System-Performance
Symptome:

Hohe CPU-Last
Verzögerungen bei Bildverarbeitung
Wachsende Datenbank

Diagnose:
bash# System-Load
top -p $(pgrep -f run_live_detection)

# Datenbank-Größe
psql -U aiuser -d ai_detection -c "
    SELECT 
        schemaname,
        tablename, 
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename))
    FROM pg_tables 
    WHERE schemaname = 'public';"
Lösungen:

CPU-Last:

Kleineres YOLO-Modell (yolov8n.pt statt yolov8x.pt)
poll_interval erhöhen
GPU-Beschleunigung aktivieren


Datenbank:

Alte Einträge löschen:



sql    DELETE FROM live_detections 
    WHERE timestamp < NOW() - INTERVAL '7 days';
    
    VACUUM ANALYZE;

Partitionierung implementieren
Archivierung einrichten


Debug-Tipps
Verbose Logging aktivieren
bash# Live-Detection
python3 run_live_detection.py --verbose

# Bewegungsanalyse (bereits aktiv)
# Logs in run_time_series_analysis.py
Pattern-Erkennung debuggen
python# In MovementDetector.py sind bereits ausführliche Logs vorhanden:
print(f"[MOVEMENT] detect_movements() mit {len(detections)} Detections")
print(f"[MOVEMENT]   Entry-Versuch {attempt + 1}: X={len(available_x)}")
print(f"[MOVEMENT]     Analysiere Y→X Sequenzen für Entry...")
print(f"[MOVEMENT]     Pattern Confidence: {confidence:.2f}")
Detection-Visualisierung
python# Bounding Boxes auf Bild zeichnen
import cv2

def visualize_detection(image_path, detection_result):
    img = cv2.imread(image_path)
    for person in detection_result['persons']:
        bbox = person['bbox']
        cv2.rectangle(img, 
                     (int(bbox[0]), int(bbox[1])),
                     (int(bbox[2]), int(bbox[3])),
                     (0, 255, 0), 2)
        cv2.putText(img, f"{person['confidence']:.2f}",
                   (int(bbox[0]), int(bbox[1])-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite('debug_detection.jpg', img)

Best Practices
1. Systemwartung
Tägliche Aufgaben:

Raumzustand überprüfen
Fehler-Logs durchsehen
Datenbank-Performance prüfen

Wöchentliche Aufgaben:

Alte Datenbank-Einträge archivieren
Statistiken analysieren
Kamera-Positionen überprüfen

Monatliche Aufgaben:

Vollständiger Datenbank-Backup
YOLO-Modell-Update prüfen
System-Performance-Analyse


2. Monitoring
Wichtige Metriken:

Detections pro Minute (beide Kameras)
Durchschnittliche Detection-Confidence
Erkannte Bewegungen pro Stunde
Raumzustands-Änderungsrate
Unverarbeitete Detections

Überwachungs-Query:
sql-- Dashboard-Query
WITH recent_stats AS (
    SELECT 
        COUNT(*) FILTER (WHERE source = 'input_x') as x_count,
        COUNT(*) FILTER (WHERE source = 'input_y') as y_count,
        AVG(avg_confidence) as avg_conf,
        COUNT(*) FILTER (WHERE processed = FALSE) as unprocessed
    FROM live_detections
    WHERE timestamp > NOW() - INTERVAL '5 minutes'
),
movements_stats AS (
    SELECT 
        COUNT(*) FILTER (WHERE movement_type = 'entry') as entries,
        COUNT(*) FILTER (WHERE movement_type = 'exit') as exits,
        AVG(confidence_score) as avg_movement_conf
    FROM movement_tracking
    WHERE timestamp > NOW() - INTERVAL '1 hour'
),
current_state AS (
    SELECT total_persons
    FROM room_state
    ORDER BY timestamp DESC
    LIMIT 1
)
SELECT * FROM recent_stats, movements_stats, current_state;

3. Fehlerbehandlung
Defensive Programmierung:

Alle Datenbank-Operationen mit Try-Catch
Plausibilitäts-Checks vor State-Updates
Graceful Degradation bei Fehlern
Detailliertes Logging

Beispiel:
pythontry:
    movement_id = self.db.insert_movement(...)
    if movement_id:
        updated = self.occupancy_manager.process_movement(...)
        if not updated:
            logger.warning("Raumzustand nicht aktualisiert")
    else:
        logger.error("Movement-Speicherung fehlgeschlagen")
except Exception as e:
    logger.error(f"Fehler bei Bewegungsverarbeitung: {e}")
    # System läuft weiter, nur dieser Durchlauf fehlgeschlagen
```

---

## Erweiterungsmöglichkeiten

### Zukünftige Features

1. **Web-Dashboard**
   - Echtzeit-Visualisierung
   - Historische Diagramme
   - Manuelle Korrektur-Interface

2. **Multi-Room-Support**
   - Mehrere Räume gleichzeitig
   - Raum-zu-Raum-Bewegungen

3. **Erweiterte Analytik**
   - Verweildauer-Analyse
   - Peak-Zeiten-Erkennung
   - Predictive Analytics

4. **Alerting**
   - Überfüllungs-Warnungen
   - Anomalie-Erkennung
   - Email/SMS-Benachrichtigungen

5. **KI-Verbesserungen**
   - Person Re-Identification
   - Gruppengrößen-Schätzung
   - Adaptive Confidence-Schwellwerte

---

## Anhang

### Dateistruktur
```
projekt/
├── BaseDetector.py                 # Abstrakte Detector-Klasse
├── UltralyticsPersonDetector.py    # YOLO-Implementation
├── DataLoader.py                   # Ordnerüberwachung
├── LiveProcessor.py                # Live-Verarbeitung
├── DatabaseHandler.py              # Datenbank-Operationen
├── MovementDetector.py             # Bewegungserkennung
├── RoomOccupancyManager.py         # Raumzustands-Verwaltung
├── TimeSeriesAnalyzer.py           # Zeitreihenanalyse
├── run_live_detection.py           # Hauptprogramm Detection
├── run_time_series_analysis.py     # Hauptprogramm Analyse
├── start_system.sh                 # System-Control-Script
├── Camera.ps1                      # Kamera-Capture (Windows)
├── input_x/                        # Eingabeordner Kamera X
├── input_y/                        # Eingabeordner Kamera Y
├── logs/                           # Log-Dateien
│   ├── detection.log
│   └── analysis.log
└── yolov8n.pt                      # YOLO-Modell

Lizenz und Autoren
Entwickelt für: DHBW Sit-Check Projekt
Version: 1.0
Datum: Januar 2025

Support
Bei Fragen oder Problemen:

Logs überprüfen (logs/detection.log, logs/analysis.log)
Datenbank-Status prüfen (siehe Abschnitt "Verwendung")
Diese Dokumentation konsultieren
Systemadministrator kontaktieren