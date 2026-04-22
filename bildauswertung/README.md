# Bildauswertung – Technische Dokumentation

**Echtzeit-Personenzählung und Raumauslastungserfassung mittels Computer Vision**

---

## Inhaltsverzeichnis

1. [Systemübersicht](#1-systemübersicht)
2. [Architekturdiagramm](#2-architekturdiagramm)
3. [Verzeichnisstruktur](#3-verzeichnisstruktur)
4. [Datenfluss: Frame → Ergebnis](#4-datenfluss-frame--ergebnis)
5. [Module im Detail](#5-module-im-detail)
   - [VideoInputModule](#51-videoinputmodule)
   - [YOLOTrackingModule](#52-yolotrackingmodule)
   - [Detektoren (LocalYOLO / API / Hybrid)](#53-detektoren)
   - [TrajectoryEntryAnalysisModule](#54-trajectoryentryanalysismodule)
   - [OccupancyStateModule](#55-occupancystatemodule)
   - [VisualizationOutputModule](#56-visualizationoutputmodule)
   - [LiveProcessor](#57-liveprocessor)
   - [Dashboard (realtime/)](#58-dashboard-realtime)
   - [Integrations-Schicht](#59-integrations-schicht)
   - [Simulationssystem](#510-simulationssystem)
6. [Zonensystem und Geometrie](#6-zonensystem-und-geometrie)
7. [Erkennungsalgorithmen](#7-erkennungsalgorithmen)
8. [Hybrid-Detektor-Strategie](#8-hybrid-detektor-strategie)
9. [Stabilisierung und Re-Identifikation](#9-stabilisierung-und-re-identifikation)
10. [Konfiguration (config.yaml)](#10-konfiguration-configyaml)
11. [Performance-Architektur](#11-performance-architektur)

---

## 1. Systemübersicht

Die Bildauswertung ist die KI-Kernkomponente des SitCheck-Systems. Sie verarbeitet einen kontinuierlichen Videostrom und erkennt in Echtzeit, wenn Personen einen definierten Eingangsbereich betreten oder verlassen. Die daraus ermittelte Raumauslastung wird sowohl lokal visualisiert als auch an das SitCheck-Backend (Prognose-Datenbank) übertragen.

### Kernaufgaben

| Aufgabe | Beschreibung |
|---|---|
| **Personendetektion** | Erkennung von Personen im Videobild mittels YOLO-Modell (lokal oder Cloud-API) |
| **Personenverfolgung** | Zuweisung persistenter IDs über mehrere Frames (Tracking) |
| **Ein-/Austritts-Erkennung** | Geometrische Analyse von Bewegungstrajektorien durch konfigurierbare Zonen |
| **Raumauslastung** | Akkumulation von Eintritten/Austritten zu aktuellem Personenbestand |
| **Echtzeit-Dashboard** | MJPEG-Videostream + Statistiken über Webbrowser |
| **Datenintegration** | Persistente Speicherung in PostgreSQL mit Spool-Fallback |

---

## 2. Architekturdiagramm

```
╔══════════════════════════════════════════════════════════════════════════╗
║                         BILDAUSWERTUNG PIPELINE                         ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │  EINGABE                                                         │    ║
║  │                                                                  │    ║
║  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐│    ║
║  │  │ Webcam/RTSP  │   │  Videodatei  │   │  YouTube (yt-dlp)    ││    ║
║  │  └──────┬───────┘   └──────┬───────┘   └──────────┬───────────┘│    ║
║  │         └──────────────────┴─────────────────────┘             │    ║
║  │                     VideoInputModule                            │    ║
║  └────────────────────────────┬────────────────────────────────────┘    ║
║                                │  BGR-Frame (H×W×3)                      ║
║  ┌─────────────────────────────▼────────────────────────────────────┐   ║
║  │  DETEKTION & TRACKING                                            │   ║
║  │                                                                  │   ║
║  │   YOLOTrackingModule                                             │   ║
║  │   ┌────────────────────────────────────────────────────────┐    │   ║
║  │   │           HybridPersonDetector                         │    │   ║
║  │   │                                                        │    │   ║
║  │   │   ┌──────────────────┐    ┌───────────────────────┐   │    │   ║
║  │   │   │  Cloud-API       │    │  Lokales YOLO-Modell   │   │    │   ║
║  │   │   │  (async Thread)  │    │  (yolo28n.pt, CPU)    │   │    │   ║
║  │   │   │  10 fps max      │    │  10 fps max           │   │    │   ║
║  │   │   └────────┬─────────┘    └──────────┬────────────┘   │    │   ║
║  │   │            └──────────────────────────┘                │    │   ║
║  │   │                   Cache-Fallback (<4 Frames)           │    │   ║
║  │   └────────────────────────────────────────────────────────┘    │   ║
║  │                                                                  │   ║
║  │   Stabilisierung: EMA-Glättung, Stale-Tracks, Bewegungsvektoren │   ║
║  └────────────────────────────┬─────────────────────────────────────┘  ║
║                                │  tracks: [{id, bbox, center, trail}]   ║
║  ┌─────────────────────────────▼────────────────────────────────────┐   ║
║  │  EIN-/AUSTRITTS-ANALYSE                                          │   ║
║  │                                                                  │   ║
║  │   TrajectoryEntryAnalysisModule                                  │   ║
║  │                                                                  │   ║
║  │   ┌────────────────┐  ┌──────────────────┐  ┌────────────────┐  │   ║
║  │   │   LINE-Modus   │  │  POLYGON-Modus   │  │ DUAL_POLYGON   │  │   ║
║  │   │  Vorzeichenwechsel│ │Punkt-in-Polygon  │  │ 2-Zonen-      │  │   ║
║  │   │   auf Normalen  │  │  Ray-Casting     │  │ Übergangs-    │  │   ║
║  │   │                │  │                  │  │ Erkennung     │  │   ║
║  │   └────────────────┘  └──────────────────┘  └────────────────┘  │   ║
║  │                        Re-ID / Trajektorien-Stitching            │   ║
║  └────────────────────────────┬─────────────────────────────────────┘  ║
║                                │  events: [{type, track_id, confidence}] ║
║  ┌─────────────────────────────▼────────────────────────────────────┐   ║
║  │  ZUSTANDSVERWALTUNG                                              │   ║
║  │  OccupancyStateModule: occupancy ±1, Deduplication              │   ║
║  └─────┬─────────────────────────────────────────────────────┬──────┘  ║
║        │                                                       │         ║
║  ┌─────▼──────────────┐                     ┌─────────────────▼──────┐  ║
║  │  VISUALISIERUNG    │                     │  INTEGRATION           │  ║
║  │                    │                     │                        │  ║
║  │  VisualizationOutput│                    │  PrognoseDbWriter      │  ║
║  │  - Bounding Boxes  │                     │  - counts-Tabelle      │  ║
║  │  - Zonen-Overlay   │                     │  - Quality Scoring     │  ║
║  │  - Zonen-Editor    │                     │  - JSONL-Spool         │  ║
║  │  - MJPEG-Stream    │                     │  - API-Ingest          │  ║
║  └────────────────────┘                     └────────────────────────┘  ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 3. Verzeichnisstruktur

```
bildauswertung/
│
├── LiveProcessor.py                  # Haupt-Orchestrator (Event-Loop)
├── ConfigManager.py                  # YAML-Konfiguration + ENV-Overrides
├── DetectorFactory.py                # Instanziierung des richtigen Detektors
│
├── # ── DETEKTION ─────────────────────────────────────────────────────────
├── BaseDetector.py                   # Abstrakte Basisklasse für alle Detektoren
├── LocalYoloPersonDetector.py        # Lokale YOLO-Inferenz (Ultralytics)
├── UltralyticsPersonDetector.py      # Cloud-API-Detektor (HTTP POST)
├── HybridPersonDetector.py           # Intelligentes Async-API + Local Switching
│
├── # ── TRACKING & ANALYSE ────────────────────────────────────────────────
├── YOLOTrackingModule.py             # EMA-Stabilisierung, Stale-Track-Handling
├── TrajectoryEntryAnalysisModule.py  # Geometrische Ein-/Austritts-Erkennung
├── OccupancyStateModule.py           # Raumauslastungs-Zustandsmaschine
│
├── # ── I/O ────────────────────────────────────────────────────────────────
├── VideoInputModule.py               # Videoqellen-Abstraktion (OpenCV + yt-dlp)
├── VisualizationOutputModule.py      # OpenCV-Rendering + interaktiver Zonen-Editor
├── DatabaseHandler.py                # PostgreSQL (tracking_events, room_state)
│
├── # ── INTEGRATION ────────────────────────────────────────────────────────
├── integration/
│   ├── prognose_db_writer.py         # Schreibt in prognose.counts
│   ├── evidence_builder.py           # Serialisiert Qualitäts-Metadaten
│   ├── quality_mapping.py            # Berechnet Quality-Score (0–1)
│   └── profile_occupancy_simulation.py  # Excel-basierte Auslastungs-Simulation
│
├── # ── DASHBOARD ──────────────────────────────────────────────────────────
├── realtime/
│   ├── dashboard_app.py              # Flask-App, MJPEG-Stream, Zone-Editor-API
│   └── simulation_remote_app.py      # Fernsteuerung der Simulationsclips
│
├── # ── KONFIGURATION ──────────────────────────────────────────────────────
├── config.yaml                       # Aktive Systemkonfiguration
├── bytetrack_entrance.yaml           # ByteTrack-Tracker-Parameter
├── models/
│   └── yolo26n.pt                    # Lokales YOLO-Nano-Modell
│
└── LiveFeed Simulation/              # Simulationsvideos (.mov / .mp4)
    ├── Leerlauf.mov
    └── ...
```

---

## 4. Datenfluss: Frame → Ergebnis

Der gesamte Verarbeitungsweg eines einzelnen Videobildes:

```
VideoInputModule.read()
        │
        │  BGR-Frame (numpy array H×W×3)
        ▼
YOLOTrackingModule.track(frame)
        │
        ├─► [Optional] Preprocessing
        │     - Upscaling (z.B. ×1.5)
        │     - CLAHE Kontrastverbesserung
        │     - Bildrauschen-Reduktion
        │
        ├─► HybridPersonDetector.track(frame)
        │     │
        │     ├─ Schritt 1: Fertiges API-Ergebnis abholen (Future)
        │     ├─ Schritt 2: Neuen API-Request starten (async, wenn Intervall ok)
        │     ├─ Schritt 3: Lokales YOLO ausführen (sync, wenn Budget ok)
        │     └─ Schritt 4: Cache zurückgeben (wenn Alter < 4 Frames)
        │
        │     Rückgabe: [{bbox, center, confidence, track_id}]
        │
        └─► Stabilisierung
              - Bounding-Box-Glättung via EMA (α=0.78)
              - Stale-Track-Holding (12 Frames, Konfidenz-Decay)
              - Bewegungsvektor-Berechnung
              - Trail-Punkte (letzte 16 Positionen)

        Rückgabe: [{track_id, bbox, center, confidence,
                    motion_vector, trail, is_stale}]
        │
        ▼
TrajectoryEntryAnalysisModule.update(tracks, frame_shape)
        │
        ├─► Track-Historien aktualisieren (deque, maxlen=40 Punkte)
        │
        ├─► Re-ID: Verschwundene Tracks wiedererkennen
        │     - Velocity-basierte Positionsvorhersage
        │     - Score = 0.74×Vorhersage-Abstand + 0.26×Letzter-Abstand
        │     - Akzeptanz: Score < dynamischer Schwellwert, keine Ambiguität
        │
        ├─► Zonenbasierte Kreuzungsdetektion:
        │     ├─ LINE: Vorzeichenwechsel der Normaldistanz
        │     ├─ POLYGON: Ray-Casting Außen→Innen
        │     └─ DUAL_POLYGON: Zonen-Übergangs-Zustandsmaschine
        │
        └─ Rückgabe: [{type: "entry"/"exit", track_id,
                        trajectory, confidence, reason}]
        │
        ▼
OccupancyStateModule.handle_event(event)
        │
        ├─► Deduplication (pro Track-ID nur 1 Eintritt/Austritt)
        ├─► Occupancy ±1
        └─► Optional: DB-Insert (tracking_events, room_state)

        Rückgabe: occupancy (int)
        │
        ├──────────────────────────────────────────────────────┐
        ▼                                                       ▼
VisualizationOutputModule.draw(...)               PrognoseDbWriter.write_frame(...)
  - Bounding Boxes (grün=Eintritt, rot=Austritt)     - Rate-Limiting (2/s)
  - Bewegungspfeile + Trails                         - Quality-Score berechnen
  - Zonen-Overlay (Polygone / Linie)                 - Evidence-JSON bauen
  - Auslastungstext                                   - INSERT INTO counts
  - cv2.imshow() oder MJPEG-Stream                   - Bei Fehler: JSONL-Spool
```

---

## 5. Module im Detail

### 5.1 VideoInputModule

**Datei:** `VideoInputModule.py`

Abstraktion über alle Videoeingangsquellen. Kapselt OpenCV `VideoCapture` und erweitert sie um YouTube-Auflösung, Reconnect-Logik und FPS-Drosselung für Datei-Quellen.

**Unterstützte Quellen:**

| `input_mode` | Beschreibung |
|---|---|
| `livefeed_simulation` | Spielt lokale Videodateien in Schleife ab |
| `file` | Einzelne lokale Videodatei |
| `youtube` | YouTube-Stream via `yt-dlp` aufgelöst |
| `rtsp` | IP-Kamera / RTSP-Stream |
| `webcam` | Lokale Kamera (Gerät `0`, `1`, ...) |

**Besonderheiten:**
- **Hardware-Acceleration (`hwaccel: auto`):** Nutzt automatisch CUDA, VideoToolbox (macOS) oder DXVA2 (Windows) für H.264-Dekodierung, sofern verfügbar.
- **FPS-Drosselung für Datei-Quellen:** Liest die FPS-Metadaten der Datei (`cv2.CAP_PROP_FPS`) und wartet zwischen Frames, um die Originalgeschwindigkeit zu simulieren. Schläft in 20ms-Schritten um den Haupt-Thread nicht zu blockieren.
- **Reconnect-Logik:** Bei Verbindungsabbruch (RTSP) wird bis zu `max_retries` Mal mit `reconnect_delay` Sekunden Pause neu verbunden. `max_retries: 0` = unbegrenzt.
- **YouTube-Cookie-Support:** Für altersgeschützte Inhalte; Cookies aus Browser oder Datei.

```python
# Kerninterface
video = VideoInputModule(source="0", hwaccel="auto")
video.open()
ok, frame = video.read()   # Gibt (bool, np.ndarray | None) zurück
video.release()
```

---

### 5.2 YOLOTrackingModule

**Datei:** `YOLOTrackingModule.py`

Wrapper um den eigentlichen Detektor. Fügt **Stabilisierung** der Bounding-Boxes und persistente **Trajectorie-Führung** hinzu.

**Stabilisierungsalgorithmus (EMA):**

Bounding-Boxes von YOLO-Modellen „zittern" von Frame zu Frame. Der YOLOTrackingModule glättet dies mit einem **Exponential Moving Average**:

```
geglättete_bbox = α × neue_bbox + (1 - α) × letzte_bbox
```

Mit `box_ema_alpha: 0.78` wird jeder neue Frame zu 78% gewichtet – schnell genug für echte Bewegungen, langsam genug um Rauschen zu dämpfen.

**Stale-Track-Handling:**

Wenn ein Track für einige Frames nicht mehr erkannt wird (Verdeckung, Modell-Fehlläufer), wird er **gehalten** statt sofort gelöscht:

- `track_hold_frames: 12` – Haltedauer in Frames
- `hold_confidence_decay: 0.9` – Konfidenz pro Frame reduziert (12 Frames: 0.9^12 ≈ 0.28)
- Danach erst endgültig entfernt

**Trail (Bewegungsspur):**
- Speichert die letzten `trail_length: 16` Zentropositionen
- Wird für Visualisierung und Geometrie-Analyse genutzt

**Rückgabe pro Track:**
```python
{
    "track_id": 42,
    "bbox": [x1, y1, x2, y2],          # Pixel, geglättet
    "center": (cx, cy),                  # Mittelpunkt
    "confidence": 0.87,                  # YOLO-Konfidenz
    "motion_vector": (dx, dy),           # Pixel/Frame
    "trail": [(cx1,cy1), (cx2,cy2)...], # Letzte 16 Positionen
    "is_stale": False                    # Wird gehalten, nicht gesehen
}
```

---

### 5.3 Detektoren

#### BaseDetector (Abstrakte Basisklasse)

**Datei:** `BaseDetector.py`

Definiert das Interface für alle Detektoren:

```python
class BaseDetector:
    def track(frame, tracker, conf, iou, imgsz, ...) -> List[Dict]
    def detect(image) -> Dict
    def get_model_info() -> Dict
```

#### LocalYoloPersonDetector

**Datei:** `LocalYoloPersonDetector.py`

Führt das YOLO-Modell lokal aus (Ultralytics-Bibliothek, Datei `models/yolo28n.pt`).

**Tracking via IOU + Distanz:**

Da lokale YOLO-Modelle keine eingebaute persistente Track-ID haben, implementiert `LocalYoloPersonDetector` ein eigenes leichtgewichtiges Tracking:

```
Für jede neue Detektion (sortiert nach Konfidenz):
  1. IOU mit allen bekannten Tracks berechnen
  2. Euklidische Distanz zum Zentrum berechnen
  3. Dynamischen Maximalabstand berechnen:
       max_dist = max(35px, min(180px, Diagonale × 0.9))
  4. Wenn IOU < 0.25 UND Distanz > max_dist → kein Match
  5. Score = 0.7 × IOU + 0.3 × (1 - Distanz/max_dist)
  6. Bestes Match → Track-ID zuweisen
  7. Kein Match → neue Track-ID vergeben

Cleanup: Tracks entfernen, die >30 Frames nicht gesehen wurden
```

**Warmup:** Beim Start wird ein Dummy-Frame durch das Modell geschickt (`warmup()`), um die JIT-Kompilierung von PyTorch zu triggern und den ersten echten Frame nicht zu verzögern.

#### UltralyticsPersonDetector

**Datei:** `UltralyticsPersonDetector.py`

Schickt jeden Frame als JPEG-kodierten HTTP-POST an eine Cloud-Inference-API.

**Request-Ablauf:**

```
Frame (numpy) → JPEG-Encode (quality=85) → HTTP POST /predict
  Header: Authorization: Bearer <api_key>
  Body: multipart/form-data mit Bilddatei

Response (JSON) → Person-Detektionen extrahieren → Tracks zurückgeben
```

**Robustheit:**
- **Failure-Cooldown (`api_failure_cooldown_seconds: 2.0`):** Nach einem API-Fehler werden für 2 Sekunden keine weiteren Requests gesendet – verhindert Spam bei Netzwerkproblemen.
- **Rate-Limiting (`max_api_fps: 10.0`):** Mindestabstand zwischen Requests = 1/10 = 100ms.
- **Response-Format-Normalisierung:** Unterstützt diverse API-Response-Formate (`predictions`, `results`, `boxes`, `detections`).

**Validierungsfilter für Personen-Detektionen:**

```python
# Filtert physikalisch unplausible Boxen:
height_ratio = bbox_height / frame_height  ≥ 0.10  # Person min. 10% Bildhöhe
area_ratio   = bbox_area   / frame_area    ≥ 0.01  # Person min. 1% Bildfläche
aspect_ratio = bbox_height / bbox_width    ≥ 1.00  # Höher als breit (Person steht)
```

#### HybridPersonDetector

**Datei:** `HybridPersonDetector.py`

Intelligente Kombination aus API und lokalem Modell. Details im Abschnitt [Hybrid-Detektor-Strategie](#8-hybrid-detektor-strategie).

#### DetectorFactory

**Datei:** `DetectorFactory.py`

Factory-Methode die anhand von `detector_mode` (`api` / `local` / `hybrid`) den richtigen Detektor instanziiert und die Frame-Intervall-Parameter berechnet:

```python
# Beispiel: Hybrid mit 20fps Target, API max 10fps, Local max 10fps
default_api_refresh_every_n  = ceil(20 / 10) = 2   # API jeden 2. Frame
default_local_refresh_every_n = ceil(20 / 10) = 2   # Local jeden 2. Frame

# Konfigurierter Wert wird nie seltener als Budget erlaubt:
api_refresh_every_n = max(1, min(configured=2, default=2)) = 2
```

---

### 5.4 TrajectoryEntryAnalysisModule

**Datei:** `TrajectoryEntryAnalysisModule.py`

Das Herzstück der Logik. Analysiert Bewegungstrajektorien und entscheidet ob ein Eintritt oder Austritt stattgefunden hat.

Drei Erkennungsmodi, konfigurierbar über `zone.mode`:

#### Modus 1: LINE

Einfachste Methode. Eine Linie teilt das Bild in zwei Hälften.

```
Algorithmus:
  1. Berechne vorzeichenbehafteten Abstand des Zentrums zur Linie:
     d = (x2-x1)*(py-y1) - (y2-y1)*(px-x1)
  2. Vergleiche Vorzeichen von d im aktuellen und letzten Frame
  3. Vorzeichenwechsel = Linien-Kreuzung
  4. Richtung bestimmt ob Eintritt oder Austritt
  5. Mindest-Versatz > min_crossing_displacement_px (16px)
```

Geeignet für: Einfache, klar getrennte Eingänge ohne Gegenstrom.

#### Modus 2: POLYGON

Ein Polygon definiert den Innenbereich des Raums.

```
Algorithmus (Ray-Casting):
  1. Speichere ersten bekannten Punkt (außerhalb)
  2. Aktuellen Punkt klassifizieren:
     - Ray vom Punkt in +X-Richtung bis Unendlich
     - Zähle Schnittpunkte mit Polygon-Kanten
     - Ungerade Anzahl = innen, gerade = außen
  3. Wenn Übergang außen→innen: Eintritt
  4. Wenn Übergang innen→außen: Austritt
```

Geeignet für: Einzel-Eingang, klare Innen/Außen-Grenze.

#### Modus 3: DUAL_POLYGON (Standard im Produktivbetrieb)

Zwei getrennte Polygone: ein Eintrittspoly und ein Austrittspoly. Robust gegen Gegenstrom und Verharren an der Grenze.

Detaillierte Beschreibung im Abschnitt [Erkennungsalgorithmen](#7-erkennungsalgorithmen).

---

### 5.5 OccupancyStateModule

**Datei:** `OccupancyStateModule.py`

Einfache Zustandsmaschine die den aktuellen Personenbestand verwaltet.

```python
Zustand: {
    occupancy: int,        # Aktueller Bestand
    entries_total: int,    # Gesamte Eintritte seit Start
    exits_total: int       # Gesamte Austritte seit Start
}

handle_event(event):
    track_id = event["track_id"]
    if track_id bereits gezählt: return False  # Deduplizierung
    
    if event["type"] == "entry":
        occupancy += 1
        entries_total += 1
    elif event["type"] == "exit":
        occupancy = max(0, occupancy - 1)
        exits_total += 1
    
    DB-Insert (optional)
    return True
```

**Deduplizierung:** Jede Track-ID kann nur einmal als Eintritt und einmal als Austritt gezählt werden. Verhindert Doppelzählungen bei kurzzeitigen Erkennungsfehlern.

---

### 5.6 VisualizationOutputModule

**Datei:** `VisualizationOutputModule.py`

OpenCV-basiertes Rendering mit interaktivem **Zonen-Editor**.

**Rendering-Elemente:**

| Element | Beschreibung |
|---|---|
| **Bounding Box** | Grün bei Eintritt, Rot bei Austritt, Weiß/Grau sonst |
| **Track-ID** | Nummer oben links an der Box |
| **Bewegungspfeil** | Zeigt aktuelle Bewegungsrichtung |
| **Trail** | Linie der letzten 16 Positionen |
| **Zonen-Overlay** | Polygon/Linie in Gelb/Grün/Rot |
| **Auslastungstext** | Aktuelle Belegung oben links |

**Zonen-Editor (interaktiv):**

```
Tastatur:
  L     → Wechsel zu LINE-Modus
  P     → Wechsel zu POLYGON-Modus
  D     → Linien-Richtung umkehren
  S     → Konfiguration speichern (config.yaml)

Maus:
  Linksklick  → Punkt hinzufügen / Linienpunkt setzen
  Rechtsklick → Letzten Punkt rückgängig

Callback:
  Jede Zonenänderung löst on_zone_changed() aus
  → LiveProcessor aktualisiert TrajectoryEntryAnalysisModule live
```

---

### 5.7 LiveProcessor

**Datei:** `LiveProcessor.py`

Der zentrale Orchestrator. Verbindet alle Module und enthält die Haupt-Event-Schleife.

```python
LiveProcessor(
    detector,              # HybridPersonDetector o.ä.
    video_source,          # Quelle
    zone_config,           # EntranceZoneConfig
    tracker_config,        # bytetrack_entrance.yaml
    db_config,             # Optional: PostgreSQL
    integration_config,    # Optional: PrognoseDbWriter
    process_every_n_frames # Frame-Sampling-Rate (Standard: 1)
)

start():
    video_input.open()
    
    while running:
        ok, frame = video_input.read()
        
        # Tracking
        if frame_idx % process_every_n_frames == 0:
            tracks = tracking_module.track(frame)
            last_tracks = tracks
        else:
            tracks = last_tracks  # Cache vom letzten Frame
        
        # Analyse
        events = entry_analysis.update(tracks, frame.shape)
        
        for event in events:
            occupancy_module.handle_event(event)
        
        # Integration
        integration_writer.write_frame(occupancy, tracks, ...)
        
        # Visualisierung
        visualization.draw(frame, tracks, zone, occupancy, events)
        
        frame_idx += 1
```

---

### 5.8 Dashboard (realtime/)

**Datei:** `realtime/dashboard_app.py`

Flask-Webanwendung für Echtzeit-Monitoring im Browser.

**Architektur (Multi-Thread):**

```
Thread 1: Capture-Loop
  VideoInput.read() → analysis_queue (deque, max 128 Frames)
  Rate: capture_max_fps = 20fps
  Bei Queue voll: ältesten Frame verwerfen

Thread 2: Analysis-Loop
  Dequeueing → Detektion → Tracking → Entry-Analyse
  Dynamic-Skip: Bei Queue-Druck werden Frames übersprungen
  Rate: durch Detektor-Latenz begrenzt

Thread 3: Flask-Server
  GET /stream  → MJPEG-Video-Stream (Endlos-Multipart)
  GET /api/status → JSON-Statistiken
  POST /api/zone  → Zone aktualisieren
  POST /api/clip  → Simulationsclip wechseln
```

**Dynamic-Skip-Logik:**

```python
# Bei Queue-Überlauf wird Analyse-Rate reduziert
if dynamic_skip_enabled and queue_len > dynamic_skip_queue_threshold (40):
    pressure_ratio = queue_len / threshold
    extra_skip = int(pressure_ratio)
    effective_process_n = min(
        dynamic_skip_max_n (1),       # Max: kein Skip (=1)
        process_every_n_frames + max(1, extra_skip)
    )
```

Mit `dynamic_skip_max_n: 1` ist Skip deaktiviert – jeder Frame wird analysiert.

**MJPEG-Stream:**

Jeder analysierte Frame wird JPEG-kodiert (quality=100) und als `multipart/x-mixed-replace` gestreamt. Unterstützt beliebig viele parallele Browser-Clients.

**DASH-Streaming (optional):**

Bei `dash.enabled: true` wird der Stream zusätzlich via FFmpeg als HLS/DASH-Segment-Stream ausgegeben (für adaptive Bitrate, geringe Latenz).

---

### 5.9 Integrations-Schicht

**Datei:** `integration/prognose_db_writer.py`

Überträgt jeden verarbeiteten Frame als Datenpunkt in die prognose-Datenbank.

**Datenmodell (`counts`-Tabelle):**

```json
{
  "ts": "2026-04-22T14:30:00Z",
  "zone_id": "library-main",
  "occupancy": 45,
  "utilization": 0.45,
  "source": "vision-direct-db",
  "quality_score": 0.92,
  "quality_flags": ["OK"],
  "evidence": {
    "model": {"name": "Hybrid API+Local YOLO v1"},
    "quality": {"score": 0.92, "flags": ["API_ACTIVE"]},
    "runtime": {
      "occupancy": 45,
      "utilization": 0.45,
      "tracks": 23,
      "events_in_frame": {"entry": 1, "exit": 0}
    }
  }
}
```

**Quality-Flags:**

| Flag | Bedeutung |
|---|---|
| `OK` | Normalbetrieb, API aktiv |
| `API_SLOW` | API-Ergebnis zu alt (stale) |
| `API_COOLDOWN` | API in Fehler-Cooldown |
| `LOCAL_ONLY` | Nur lokales Modell verfügbar |
| `CACHE_STALE` | Gecachte Ergebnisse > 6 Frames alt |

**Spool-Mechanismus:**

Bei Datenbankausfall werden Records in eine lokale JSONL-Datei geschrieben. Beim nächsten erfolgreichen Schreib-Zyklus werden bis zu `flush_batch_size: 200` gespoollte Records nachgereicht.

**Rate-Limiting:** Maximal `max_writes_per_second: 2.0` Datenbankzugriffe – unabhängig vom Frame-Rate.

---

### 5.10 Simulationssystem

#### ProfileOccupancySimulation

**Datei:** `integration/profile_occupancy_simulation.py`

Simuliert Auslastungsverläufe auf Basis historischer Excel-Daten. Läuft im Hintergrund-Thread und emittiert synthetische Eintritts-/Austrittsereignisse.

**Excel-Profil-Verarbeitung:**

```
Excel-Datei (KI_Projekt_Daten_einJahr.xlsx):
  Spalten: timestamp, occupancy

Verarbeitung:
  1. Alle (timestamp, occupancy)-Paare laden
  2. In 15-Minuten-Slots gruppieren:
     slot = Stunde × 4 + Minute ÷ 15    (0–95)
     bucket = (Wochentag 0-6, slot 0-95)
  3. Pro Bucket: Mittelwert + Standardabweichung berechnen
     → 7 × 96 = 672 Buckets (ein vollständiges Jahresprofil)

Tick-Algorithmus (alle 60 Sekunden):
  target = Profil-Mittelwert[aktueller Bucket] + N(0, σ) × noise_scale
  blended = profile_blend × target + (1-profile_blend) × current
  delta   = clamp(blended - current, -max_step_per_tick, +max_step_per_tick)
  current += delta
  → Emit entsprechende Eintritts/Austritts-Ereignisse
```

#### LiveFeed Simulation

Verzeichnis `LiveFeed Simulation/` enthält voraufgezeichnete Videoclips.  
`simulation_remote_app.py` bietet eine REST-API zur Fernsteuerung (Clip-Auswahl, Play/Pause) – z.B. für Demozwecke oder Tests ohne Live-Kamera.

---

## 6. Zonensystem und Geometrie

Die Zones werden in normalisierten Koordinaten (0.0–1.0 relativ zu Bildbreite/-höhe) gespeichert, damit sie auflösungsunabhängig sind.

### Konfigurationsstruktur (EntranceZoneConfig)

```python
@dataclass
class EntranceZoneConfig:
    mode: str                           # "line" | "polygon" | "dual_polygon"
    
    # LINE-Modus
    line_p1: Tuple[float, float]        # Normalisiert (0-1)
    line_p2: Tuple[float, float]
    line_entry_direction: str           # "negative_to_positive" / "positive_to_negative"
    
    # POLYGON-Modus
    polygon_points: List[Tuple[float, float]]
    
    # DUAL_POLYGON-Modus
    entry_polygon_points: List[Tuple[float, float]]
    exit_polygon_points: List[Tuple[float, float]]
    
    min_crossing_displacement_px: float = 16.0
    min_track_points: int = 3
    min_event_cooldown_frames: int = 5
```

### Geometrische Kernoperationen

**1. Vorzeichenbehaftete Distanz zur Linie:**
```
d(P, L) = (x₂ - x₁)(Py - y₁) - (y₂ - y₁)(Px - x₁)

Positiv/Negativ → Seite der Linie
Vorzeichenwechsel → Kreuzung
```

**2. Punkt-in-Polygon (Ray-Casting):**
```
Strahl von P in +X-Richtung bis ∞
Für jede Kante des Polygons:
  - Schnittpunkt mit Strahl berechnen
  - Zählen
Ungerade Anzahl → P liegt innen
```

**3. Abstand zur Polygon-Grenze:**
```
Für jede Kante: Abstand von P zum nächsten Punkt auf der Strecke
Minimum aller Kanten = Randabstand
```

**4. Zonen-Klassifikation (Dual-Polygon):**
```
Wenn P in BEIDEN Polygonen:
    → Näheres Zentroid gewinnt
Wenn P in genau einem:
    → Dieses Polygon
Wenn P außerhalb beider, Randabstand ≤ tolerance (8px):
    → Nächstes Polygon
Sonst:
    → "none" (Niemandsland)
```

---

## 7. Erkennungsalgorithmen

### Dual-Polygon: Zustandsmaschine pro Track

```
┌─────────────────────────────────────────────────────────────────┐
│ DUAL_POLYGON: Zustandsübergänge                                 │
│                                                                 │
│  Erscheint                                                      │
│  im Bild   ──► Klassifiziere Zone ──► last_seen_primary = Zone  │
│                                                                 │
│         ┌─────────────────────────────────────────────────────┐ │
│         │ Nächste Frames:                                     │ │
│         │   zone_history.append(classify(center))             │ │
│         │   stable_zone = majority_vote(last 6)               │ │
│         │                                                     │ │
│         │   if stable_zone == last_seen_primary:              │ │
│         │       return None  ← Keine Änderung                 │ │
│         │                                                     │ │
│         │   if Übergang nicht {entry,exit} ↔ {entry,exit}:   │ │
│         │       last_seen = stable_zone                       │ │
│         │       return None  ← Ungültiger Übergang            │ │
│         └─────────────────────────────────────────────────────┘ │
│                           │                                     │
│                           ▼ Potentielle Kreuzung                │
│         ┌─────────────────────────────────────────────────────┐ │
│         │ Validierung:                                        │ │
│         │                                                     │ │
│         │ 1. Mindestversatz:                                  │ │
│         │    Δ > min_displacement × 0.20                      │ │
│         │                                                     │ │
│         │ 2. Beweis-Check (ODER-Bedingung):                   │ │
│         │    a) Teillinie überquert?                          │ │
│         │       → signed_distance Vorzeichenwechsel           │ │
│         │    b) Klarer Zonenwechsel?                          │ │
│         │       → Erste Hälfte Mehrheit ≠ Zweite Hälfte       │ │
│         │                                                     │ │
│         │ 3. Fast-Mover (2-3 Punkte):                        │ │
│         │    Direkter Zonen-Sprung + Mindestschrittgröße      │ │
│         └─────────────────────────────────────────────────────┘ │
│                           │                                     │
│                           ▼ Kreuzung bestätigt                  │
│     ┌─────────────────────────────────────────────────────┐     │
│     │ event = {                                           │     │
│     │   "type": "entry" wenn exit→entry else "exit",     │     │
│     │   "track_id": ...,                                  │     │
│     │   "confidence": 1.0 (Linie) oder 0.85 (Zonen),    │     │
│     │   "reason": "dual_polygon_line_cross_..." etc.      │     │
│     │ }                                                   │     │
│     └─────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

### Fast-Mover-Erkennung (2–3 Punkte)

Personen, die schnell durch den Eingangsbereich laufen, haben möglicherweise nur 2–3 Trajektorien-Punkte bevor sie verschwinden. Eigener Algorithmus:

```
1. Vorherige Zone ≠ aktuelle Zone → direkter Sprung
   ODER: Letzte 4 Punkte zeigen Zonenwechsel

2. Schrittgröße ≥ max(4px, min_displacement × 0.30)

3. Teillinie überquert (Vorzeichenwechsel)
   ODER: mindestens letzter+vorletzter Punkt auf verschiedenen Seiten

→ Eintritt/Austritt mit Konfidenz 0.85
```

### Lingering-Erkennung (verschwundene Tracks)

```
Wenn Track verschwindet während noch in entry-Zone:
  - Hat er sich nach unten bewegt? (Δy > Schwellwert)
  - Befindet er sich im unteren Bilddrittel?
  → Wahrscheinlicher Eintritt → Event mit Konfidenz 0.65-0.75
  
Wenn Track verschwindet in exit-Zone:
  - Hat er sich nach oben bewegt?
  - Befindet er sich im oberen Bilddrittel?
  → Wahrscheinlicher Austritt → Event mit Konfidenz 0.65-0.75
```

---

## 8. Hybrid-Detektor-Strategie

Das Kernprinzip: **Cloud-API für Qualität, lokales Modell für Verfügbarkeit, Cache als Sicherheitsnetz.**

### Ablaufdiagramm pro Frame

```
track(frame):
    frame_counter++

    ┌─────────────────────────────────────────────┐
    │ Schritt 1: API-Ergebnis abholen             │
    │   Falls async Future fertig:                │
    │   → API-Ergebnis zurückgeben (sofort)       │
    │   → Future auf None setzen                  │
    └─────────────────┬───────────────────────────┘
                      │ (kein Ergebnis)
    ┌─────────────────▼───────────────────────────┐
    │ Schritt 2: Neuen API-Request starten?       │
    │   Bedingungen (ALLE müssen erfüllt sein):   │
    │   ✓ api_detector vorhanden                  │
    │   ✓ frame_counter - last_api_frame ≥ 2      │
    │   ✓ Kein aktiver Future                     │
    │   ✓ Nicht in Fehler-Cooldown               │
    │   ✓ Zeit seit letztem Request ≥ 100ms      │
    │                                             │
    │   → ThreadPoolExecutor.submit(api_call)     │
    │   → Sofort weiter (non-blocking)            │
    └─────────────────┬───────────────────────────┘
                      │
    ┌─────────────────▼───────────────────────────┐
    │ Schritt 3: Lokales YOLO ausführen?          │
    │   Bedingungen:                              │
    │   ✓ local_fill_enabled                      │
    │   ✓ frame_counter - last_local_frame ≥ 2    │
    │   ✓ Zeit seit letztem Local-Run ≥ 100ms    │
    │                                             │
    │   (ODER: force_local wenn Cache zu alt)     │
    │                                             │
    │   → Synchron ausführen (~100ms auf CPU)     │
    │   → Ergebnis zurückgeben                    │
    └─────────────────┬───────────────────────────┘
                      │ (kein lokales Ergebnis)
    ┌─────────────────▼───────────────────────────┐
    │ Schritt 4: Cache zurückgeben                │
    │   Falls cache_age ≤ 4 Frames:              │
    │   → Gecachten Track-Stand zurückgeben       │
    │                                             │
    │   Sonst:                                    │
    │   → Leere Liste zurückgeben                 │
    └─────────────────────────────────────────────┘
```

### Frame-Timing-Analyse (20fps, API 60ms)

```
Zeit →  0ms    50ms   100ms  150ms  200ms  250ms  300ms
Frame →  1      2      3      4      5      6      7

API:  [=submit=|====60ms====|done]
      Frame 1 gesendet           Frame 3: Ergebnis abgeholt
                                 Frame 4: neuer Request

Local: [100ms] ← Frame 1
                     [100ms] ← Frame 4
                                      [100ms] ← Frame 7

Result: Local  Cache  API    Local  Cache  API    Local
              (1Fr.)        (1Fr.)        (1Fr.)
```

→ **Jeder Frame erhält ein Ergebnis** (API, Local oder maximal 1 Frame Cache)  
→ **Keine Überlastung:** API alle 100ms, Local alle 100ms, nie gleichzeitig dominant

---

## 9. Stabilisierung und Re-Identifikation

### EMA-Stabilisierung

```
Hintergrund: YOLO gibt pro Frame leicht unterschiedliche Bounding-Boxes zurück,
auch wenn die Person sich nicht bewegt (Messzittern).

Lösung: Exponential Moving Average (EMA)

  bbox_smooth(t) = α × bbox_raw(t) + (1-α) × bbox_smooth(t-1)
  
  α = box_ema_alpha = 0.78

  Interpretation:
  - Hoher α: Reagiert schnell auf echte Bewegung
  - Niedriger α: Stärkere Dämpfung, glatter aber träger

  Ergebnis bei 78%: Rauschen ~5px → <1px geglättet, echte Bewegung folgt in 2-3 Frames
```

### Track Re-Identifikation (Re-ID)

Problem: Person kurz verdeckt → YOLO vergibt neue Track-ID → doppelte Zählung.

Lösung: Wenn eine neue Track-ID mit <3 Punkten auftaucht, Identitätsprüfung:

```
Neue Track-ID erscheint:

1. Suche in letzten 24 Frames verschwundene Tracks
2. Für jeden Kandidaten:
   a. Extrapoliere letzte bekannte Position:
      predicted_pos = last_pos + velocity × frame_gap
   b. Berechne Score:
      score = 0.74 × dist(new_pos, predicted_pos)
            + 0.26 × dist(new_pos, last_known_pos)
3. Bester Kandidat:
   Akzeptiere wenn:
   - Score < dynamischer Schwellwert (120px + Aktivitäts-Marge)
   - Zweibester Score / Bester Score > 0.87 (Ambiguität-Check)
4. Bei Akzeptanz:
   - Übernehme alte Track-ID
   - Füge Historien zusammen
   - Re-ID-Event registrieren
```

Der **Ambiguität-Check** verhindert falsche Re-IDs in belebten Szenen: Wenn zwei Kandidaten ähnlich gut passen, wird kein Merge durchgeführt (lieber eine neue ID vergeben).

---

## 10. Konfiguration (config.yaml)

Die gesamte Systemkonfiguration wird in `bildauswertung/config.yaml` verwaltet. ENV-Variablen überschreiben einzelne Parameter (`SITCHECK_DETECTOR_MODE` etc.).

### Vollständige Konfigurationsreferenz

```yaml
# ── VIDEOEINGABE ────────────────────────────────────────────────────────────
video:
  source: "/sitcheck/bildauswertung/LiveFeed Simulation/Leerlauf.mov"
  fallback_source: ""          # Fallback bei Verbindungsfehler
  reconnect_delay: 1.0         # Sekunden zwischen Reconnect-Versuchen
  max_retries: 0               # 0 = unbegrenzt
  hwaccel: auto                # auto | cuda | videotoolbox | dxva2 | none
  input_mode: livefeed_simulation  # file | youtube | rtsp | livefeed_simulation
  simulation:
    directory: "LiveFeed Simulation"
    control_mode: remote_control
    default_clip_id: leerlauf
    idle_loop: false

# ── TRACKING & DETEKTION ────────────────────────────────────────────────────
tracking:
  detector_mode: hybrid        # api | local | hybrid

  # Cloud-API
  api_url: "https://..."
  api_key: "ul_..."
  api_timeout_seconds: 3.0
  api_failure_cooldown_seconds: 2.0
  max_api_fps: 10.0            # Max. API-Requests pro Sekunde
  api_jpeg_quality: 85         # JPEG-Kompression für Upload (60-95)
  api_refresh_every_n_frames: 2  # API jeden N-ten Frame anfragen

  # Lokales Modell
  model_path: models/yolo26n.pt
  device: auto                 # auto | cpu | cuda
  local_preload_enabled: true  # Warmup beim Start
  local_fill_enabled: true     # Lokales Modell als Lückenfüller
  local_fill_max_fps: 10.0     # Max. lokale Inferenzen pro Sekunde
  local_refresh_every_n_frames: 2  # Lokales Modell jeden N-ten Frame

  # Hybrid-Parameter
  hybrid_target_fps: 20.0      # Ziel-FPS für Kadenz-Berechnung
  cache_fallback_max_age_frames: 4  # Cache max. N Frames alt
  api_result_max_age_frames: 6  # API-Ergebnis max. N Frames alt verwerfen
  max_cache_only_frames: 2     # Force Local nach N Cache-only Frames

  # Detektions-Filter
  confidence_threshold: 0.2    # Mindest-Konfidenz
  iou_threshold: 0.45          # IOU für NMS
  min_person_height_ratio: 0.1   # Mindesthöhe (10% Bildhöhe)
  min_person_area_ratio: 0.01    # Mindestfläche (1% Bildfläche)
  min_person_aspect_ratio: 1.0   # Höher als breit

  # Inferenz-Parameter
  tracker: bytetrack_entrance.yaml
  imgsz: 640                   # YOLO-Eingabegröße
  max_detections: 300

  # Stabilisierung
  stabilization_enabled: true
  track_hold_frames: 12        # Frames Stale-Track-Holding
  box_ema_alpha: 0.78          # EMA-Glättungs-Faktor
  hold_confidence_decay: 0.9   # Konfidenz-Abbau pro Frame
  trail_length: 16             # Länge der Bewegungsspur
  motion_min_pixels: 1.2       # Mindestbewegung für Vektoranzeige
  process_every_n_frames: 1    # Tracking jeden N-ten Frame

  # Re-Identifikation
  reid_max_gap_frames: 24      # Max. Frame-Lücke für Re-ID
  reid_max_distance_px: 150.0  # Max. Distanz für Re-ID
  reid_ambiguity_ratio: 0.87   # Ambiguität-Schwellwert

# ── VORVERARBEITUNG ─────────────────────────────────────────────────────────
preprocess:
  enabled: false
  upscale: 1.0                 # Hochskalierungsfaktor (z.B. 1.5)
  clahe_clip: 2.0              # CLAHE Kontrastlimit
  denoise: false               # Bildrauschen-Reduktion

# ── DASHBOARD ───────────────────────────────────────────────────────────────
dashboard:
  stream_fps: 20               # Ziel-FPS für Capture und Stream
  capture_max_fps: 20
  visual_update_fps: 20
  jpeg_quality: 100            # Stream-JPEG-Qualität
  stream_max_width: 1280       # Max. Stream-Breite (px)

  analysis_queue_frames: 128   # Max. Frames in Analyse-Queue
  analysis_skip_threshold_frames: 60  # Frames droppen wenn Queue > N
  dynamic_skip_enabled: true
  dynamic_skip_queue_threshold: 40  # Ab hier adaptive Skip-Rate
  dynamic_skip_max_n: 1        # Max. Skip-Faktor (1=kein Skip)

  profile_simulation:
    enabled: true
    excel_path: KI_Projekt_Daten_einJahr.xlsx
    tick_seconds: 60
    profile_blend: 0.72
    noise_sigma_scale: 0.85
    max_step_per_tick: 2.0
    rollback_minutes: 15.0

  dash:                        # HLS/DASH-Streaming via FFmpeg
    enabled: true
    output_dir: runtime/dash
    segment_time: 0.5          # Sekunden pro HLS-Segment
    preset: ultrafast
    tune: zerolatency
    crf: 36
    abr:
      enabled: true
      high_bitrate_kbps: 1400
      low_bitrate_kbps: 650

# ── ZONEN-KONFIGURATION ─────────────────────────────────────────────────────
zone:
  mode: dual_polygon           # line | polygon | dual_polygon

  line:
    p1: [0.94, 0.53]           # Normalisiert 0.0-1.0
    p2: [0.94, 0.66]
    entry_direction: positive_to_negative

  entry_polygon:               # Eingangszone (z.B. linke Seite)
    points: [[0.01, 0.02], [0.01, 0.99], [0.52, 0.99], [0.51, 0.01]]

  exit_polygon:                # Ausgangszone (z.B. rechte Seite)
    points: [[0.51, 0.00], [0.53, 0.99], [1.00, 0.99], [0.99, 0.00]]

  min_crossing_displacement_px: 16.0
  min_track_points: 3
  min_event_cooldown_frames: 5

# ── BENUTZEROBERFLÄCHE ──────────────────────────────────────────────────────
ui:
  show_window: true            # OpenCV-Fenster anzeigen
  enable_zone_editor: true
  window_name: "Library Entry Tracking"

# ── LOKALE DATENBANK (optional) ─────────────────────────────────────────────
database:
  enabled: false
  host: localhost
  port: 5432
  user: aiuser
  password: "..."
  database: ai_detection

# ── PROGNOSE-INTEGRATION ────────────────────────────────────────────────────
integration:
  prognose_db:
    enabled: true
    api_ingest_enabled: true   # REST-API statt direktes DB
    api_base_url: "http://127.0.0.1:8000"
    api_timeout_seconds: 3.0
    zone_id: default-zone
    source: vision-direct-db
    max_writes_per_second: 2.0
    default_zone_capacity: 100
    capacity_refresh_seconds: 30
    flush_batch_size: 200      # Spool-Batch-Größe
    spool:
      enabled: true
      path: "../website-dashboard/runtime/logs/prognose_counts_spool.jsonl"
      max_entries: 20000
```

---

## 11. Performance-Architektur

### Frame-Verarbeitungs-Budget (20fps Ziel)

```
Verfügbare Zeit pro Frame: 1000ms / 20fps = 50ms

┌─────────────────────────────────────────────────────────────────┐
│ Operation              │ Laufzeit   │ Thread   │ Blocking?      │
├─────────────────────────────────────────────────────────────────┤
│ VideoInput.read()      │  <5ms      │ Capture  │ Ja (kurz)      │
│ JPEG-Encode (API)      │  2-5ms     │ Analysis │ Ja             │
│ HTTP POST (API)        │  ~60ms     │ Async    │ Nein (Future)  │
│ YOLO lokal (CPU)       │  ~100ms    │ Analysis │ Ja             │
│ EMA-Stabilisierung     │  <1ms      │ Analysis │ Ja             │
│ Geometrie-Analyse      │  <2ms      │ Analysis │ Ja             │
│ JPEG-Encode (Stream)   │  5-15ms    │ Analysis │ Ja             │
│ DB-Write (async)       │  <10ms     │ Writer   │ Nein           │
└─────────────────────────────────────────────────────────────────┘

Analysis-Thread-Auslastung pro Frame:
  → Wenn Local YOLO:   ~107ms  (überschreitet 50ms → Dynamic Skip greift)
  → Wenn API-Cache:     ~7ms   (weit unter 50ms)
  → Durchschnitt:     ~57ms   (≈ 17-19fps effektiv)
```

### Ressourcennutzung

| Ressource | Wert | Konfiguriert durch |
|---|---|---|
| CPU (YOLO lokal) | ~100ms/Frame | `device: cpu`, Modellgröße |
| CPU (Gesamt) | 20-40% (1 Kern) | abhängig von detection_mode |
| RAM | ~500MB–1GB | YOLO-Modell im Speicher |
| GPU (optional) | ~5-20ms/Frame | `device: cuda` |
| Netzwerk (API) | ~50-200 KB/Request | `api_jpeg_quality: 85` |
| Speicher (Spool) | max. ~20MB | `spool_max_entries: 20000` |

### Skalierungsverhalten

```
detector_mode: local
  + Keine Netzwerkabhängigkeit
  + Kein API-Kosten
  - Langsamer (CPU ~100ms/Frame)
  
detector_mode: api
  + Schnell (~60ms, GPU in Cloud)
  + Kein lokaler GPU nötig
  - Netzwerkabhängigkeit
  - API-Kosten

detector_mode: hybrid (Standard)
  + Ausfallsicher (API-Fehler → Local → Cache)
  + Beste Abdeckung aller Frames
  + Optimale CPU-Auslastung (API und Local wechselnd)
  - Komplexestes Timing
```

### Fehlertoleranz-Matrix

| Ausfall | Verhalten |
|---|---|
| API nicht erreichbar | Cooldown 2s, dann Local-only |
| Lokales Modell langsam | Cache bis 4 Frames, dann erzwungener Local-Lauf |
| Kamera-Verbindungsabbruch | Reconnect-Loop mit konfigurierbarer Retry-Verzögerung |
| Datenbankausfall | JSONL-Spool, Nachlieferung beim nächsten erfolgreichen Write |
| Queue-Überlauf | Älteste Frames verworfen, Dynamic-Skip reduziert Analyse-Rate |

---

*Dokumentation erstellt für die SitCheck-Systempräsentation | Bildauswertung v2.x | April 2026*
