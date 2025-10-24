# Live-Personenerkennung mit Zeitreihenanalyse

Ein vereinfachtes System zur Echtzeit-Personenerkennung mit Ultralytics YOLO, das zwei Ordner überwacht und die Ergebnisse mit Zeitreihenanalyse korreliert.

## 🎯 Funktionsweise

Das System besteht aus zwei Hauptkomponenten:

1. **Live-Detection**: Überwacht kontinuierlich zwei Ordner (`input_x` und `input_y`) auf neue Bilder
   - Bilder werden automatisch erkannt und geladen
   - Ultralytics YOLO führt Personenerkennung durch
   - Ergebnisse werden sofort in PostgreSQL gespeichert
   - Bilder werden nach Verarbeitung gelöscht (Pipeline-Prinzip)

2. **Zeitreihenanalyse**: Korreliert die Detections aus beiden Ordnern
   - Paart zeitlich nahe Detections (max. 5 Sekunden Differenz)
   - Schätzt die tatsächliche Personenanzahl durch verschiedene Strategien
   - Speichert bereinigte Ergebnisse in separater Tabelle

## 📋 Voraussetzungen

### Software
- Python 3.8+
- PostgreSQL 12+

### Python-Pakete
```bash
pip install ultralytics opencv-python pg8000 numpy
```

## 🗄️ Datenbankstruktur

### Tabelle: `live_detections`
Speichert Rohdaten von beiden Ordnern:
- `id`: Primärschlüssel
- `timestamp`: Zeitstempel der Detection
- `source`: Ordner-Name (`input_x` oder `input_y`)
- `persons_detected`: Anzahl erkannter Personen
- `avg_confidence`, `max_confidence`, `min_confidence`: Konfidenzwerte
- `detection_data`: Vollständige Detection-Daten (JSONB)

### Tabelle: `correlated_persons`
Speichert korrelierte/bereinigte Ergebnisse:
- `id`: Primärschlüssel
- `timestamp`: Zeitstempel der Korrelation
- `source_x_id`, `source_y_id`: Referenzen zu `live_detections`
- `persons_x`, `persons_y`: Rohdaten aus beiden Quellen
- `estimated_actual_persons`: Geschätzte tatsächliche Personenanzahl
- `confidence_score`: Konfidenz der Schätzung
- `time_diff_seconds`: Zeitdifferenz zwischen Detections
- `analysis_data`: Analyse-Details (JSONB)

## 🚀 Verwendung

### 1. Datenbank vorbereiten
```bash
# Mit dem Setup-Script (automatisch)
cat > setup_database.sh << 'EOF'
#!/bin/bash
sudo -u postgres psql << PSQL
DO \$\$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_user WHERE usename = 'aiuser') THEN
        CREATE USER aiuser WITH PASSWORD 'DHBW1234!?';
    END IF;
END \$\$;

SELECT 'CREATE DATABASE ai_detection OWNER aiuser'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'ai_detection')\gexec

GRANT ALL PRIVILEGES ON DATABASE ai_detection TO aiuser;
\c ai_detection
GRANT ALL ON SCHEMA public TO aiuser;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO aiuser;
PSQL
EOF

chmod +x setup_database.sh
./setup_database.sh
```

### 2. Konfiguration anpassen
Bearbeite `start_system.sh` und passe die Datenbank-Credentials an:
```bash
DB_HOST="localhost"
DB_USER="aiuser"
DB_PASSWORD="DHBW1234!?"
DB_NAME="ai_detection"
```

### 3. System starten
```bash
chmod +x start_system.sh
./start_system.sh start
```

### 4. Bilder hinzufügen
Kopiere Bilder in die Ordner:
```bash
# Beispiel
cp bild1.jpg input_x/
cp bild2.jpg input_y/
```

Die Bilder werden automatisch erkannt, verarbeitet und gelöscht.

### 5. Logs verfolgen
```bash
./start_system.sh logs
```

### 6. Status prüfen
```bash
./start_system.sh status
```

### 7. System stoppen
```bash
./start_system.sh stop
```

## 🔧 Erweiterte Nutzung

### Nur Live-Detection starten
```bash
python3 run_live_detection.py \
    --db-user aiuser \
    --db-password "DHBW1234!?" \
    --db-name ai_detection \
    --input-x input_x \
    --input-y input_y \
    --confidence-threshold 0.5
```

### Nur Zeitreihenanalyse starten
```bash
python3 TimeSeriesAnalyzer.py \
    --db-user aiuser \
    --db-password "DHBW1234!?" \
    --db-name ai_detection \
    --interval 10
```

### Einmalige Analyse durchführen
```bash
python3 TimeSeriesAnalyzer.py \
    --db-user aiuser \
    --db-password "DHBW1234!?" \
    --db-name ai_detection \
    --once
```

### Zusammenfassung anzeigen
```bash
python3 TimeSeriesAnalyzer.py \
    --db-user aiuser \
    --db-password "DHBW1234!?" \
    --db-name ai_detection \
    --summary 24  # Letzte 24 Stunden
```

## 📊 Zeitreihenanalyse-Strategien

Die Zeitreihenanalyse verwendet verschiedene Strategien zur Schätzung der tatsächlichen Personenanzahl:

1. **Perfekte Übereinstimmung**: Wenn beide Quellen die gleiche Anzahl liefern → Wert übernehmen
2. **Große Abweichung** (>2 Personen): Maximum nehmen (konservative Schätzung)
3. **Konfidenz-basiert**: Bei signifikantem Konfidenzunterschied (>0.1) → Wert mit höherer Konfidenz
4. **Gewichteter Durchschnitt**: Bei ähnlicher Konfidenz → nach Konfidenz gewichteter Mittelwert

Zusätzlich wird eine Zeitstrafe angewendet: Je größer die Zeitdifferenz zwischen den Detections, desto geringer die Konfidenz des Ergebnisses.

## 📁 Projektstruktur

```
.
├── BaseDetector.py              # Abstrakte Basisklasse
├── UltralyticsPersonDetector.py # YOLO-Implementierung
├── DatabaseHandler.py           # PostgreSQL-Operationen
├── DataLoader.py                # Ordner-Überwachung
├── LiveProcessor.py             # Live-Verarbeitung
├── TimeSeriesAnalyzer.py        # Zeitreihenanalyse
├── run_live_detection.py        # Hauptprogramm Detection
├── start_system.sh              # System-Management-Script
├── input_x/                     # Eingabeordner X
├── input_y/                     # Eingabeordner Y
└── logs/                        # Log-Dateien
    ├── detection.log
    └── analysis.log
```

## 🔍 Monitoring

### Datenbank-Abfragen

**Letzte Detections anzeigen:**
```sql
SELECT timestamp, source, persons_detected, avg_confidence 
FROM live_detections 
ORDER BY timestamp DESC 
LIMIT 10;
```

**Korrelierte Ergebnisse anzeigen:**
```sql
SELECT 
    timestamp,
    persons_x,
    persons_y,
    estimated_actual_persons,
    confidence_score,
    time_diff_seconds
FROM correlated_persons 
ORDER BY timestamp DESC 
LIMIT 10;
```

**Statistik über letzte Stunde:**
```sql
SELECT 
    source,
    COUNT(*) as count,
    AVG(persons_detected) as avg_persons,
    AVG(avg_confidence) as avg_confidence
FROM live_detections 
WHERE timestamp >= NOW() - INTERVAL '1 hour'
GROUP BY source;
```

## ⚙️ Konfigurationsparameter

### Live-Detection
- `--poll-interval`: Prüfintervall für neue Bilder (Standard: 0.5s)
- `--confidence-threshold`: Mindest-Konfidenz für Detections (Standard: 0.5)
- `--yolo-model`: YOLO-Modell (Standard: yolov8n.pt)

### Zeitreihenanalyse
- `--interval`: Intervall zwischen Analysen (Standard: 10s)
- `max_time_diff`: Max. Zeitdifferenz für Paare (in Code: 5.0s)
- `confidence_threshold`: Mindest-Konfidenz für Korrelation (in Code: 0.5)

## 🐛 Fehlerbehebung

### "Erkannte Personen: 0" bei Testbildern
**Das ist normal!** YOLO wurde auf echte Fotos trainiert und erkennt keine:
- Strichmännchen
- Gezeichnete Bilder
- Synthetische/künstliche Bilder
- Stark vereinfachte Darstellungen

**Lösung:** Verwende echte Fotos mit Personen
```bash
# Test mit echtem Foto
python3 test_with_real_image.py --sample crowd --save-result

# Oder verwende eigene Fotos
python3 test_with_real_image.py --image mein_foto.jpg --save-result
```

### Keine Bilder werden verarbeitet
- Prüfe ob Ordner existieren: `ls -la input_x input_y`
- Prüfe Dateirechte: `chmod 755 input_x input_y`
- Prüfe Logs: `tail -f logs/detection.log`

### Datenbank-Verbindungsfehler
- Teste Verbindung: `./start_system.sh test-db`
- Prüfe PostgreSQL-Status: `systemctl status postgresql`
- Prüfe Credentials in `start_system.sh`

### YOLO-Modell nicht gefunden
- Lade Modell herunter: `yolo download yolov8n.pt` (wird automatisch beim ersten Start gemacht)
- Oder gib expliziten Pfad an: `--yolo-model /pfad/zum/modell.pt`

## 📈 Performance-Tipps

1. **GPU-Beschleunigung**: YOLO nutzt automatisch CUDA falls verfügbar
2. **Größeres Modell**: Für bessere Genauigkeit `yolov8m.pt` oder `yolov8l.pt` verwenden
3. **Poll-Intervall anpassen**: Bei wenigen Bildern Intervall erhöhen um CPU zu sparen
4. **Analyse-Intervall**: Bei hoher Last Analyse-Intervall erhöhen

## 📝 Lizenz

Dieses Projekt ist für Bildungszwecke entwickelt.