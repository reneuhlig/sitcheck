# Test-Anleitung für Live-Personenerkennung

Dieses Dokument beschreibt alle verfügbaren Tests für das System.

## 📋 Übersicht der Test-Scripts

### 1. `test_system.py` - Basis-Systemtests
Testet die grundlegende Funktionalität ohne echte Bilder.

### 2. `test_with_real_image.py` - Erweiterte Tests mit echten Fotos
Testet Detection und Zeitreihenanalyse mit echten Bildern.

---

## 🧪 Test-Szenarien

### Szenario 1: Schneller Basis-Check (EMPFOHLEN ZUM START)

Prüft ob das System grundsätzlich funktioniert:

```bash
# Datenbank + System-Check
python3 test_system.py full-test \
    --db-user aiuser \
    --db-password "DHBW1234!?"
```

**Erwartetes Ergebnis:**
- ✓ Datenbankverbindung funktioniert
- ✓ Tabellen werden erstellt
- ✓ YOLO-Modell lädt
- ⚠️ 0 Personen erkannt (normal bei synthetischen Bildern)

---

### Szenario 2: Detection mit echtem Foto testen

Testet die Personenerkennung mit einem echten Bild:

```bash
# Mit Beispielbild (wird heruntergeladen)
python3 test_with_real_image.py --sample crowd --save-result

# Mit eigenem Foto
python3 test_with_real_image.py --image mein_foto.jpg --save-result
```

**Erwartetes Ergebnis:**
```
================================================================================
🔍 PERSONENERKENNUNG TEST
================================================================================
  Bild: temp_test_images/crowd.jpg
  Konfidenz-Schwelle: 0.5
================================================================================

✓ Bild geladen: 640x427 px
🚀 Initialisiere YOLO-Modell...
✓ Modell geladen

🔍 Führe Personenerkennung durch...

================================================================================
📊 ERGEBNISSE
================================================================================
  Erkannte Personen: 5
  Durchschnittliche Konfidenz: 0.847
  Maximale Konfidenz: 0.923
  Minimale Konfidenz: 0.721

  Details zu jeder Person:
    Person 1: Konfidenz=0.923, Position=[x:120, y:80, w:140, h:320]
    Person 2: Konfidenz=0.891, Position=[x:280, y:95, w:150, h:310]
    ...

✓ Ergebnis-Bild gespeichert: temp_test_images/result_crowd.jpg
================================================================================
```

---

### Szenario 3: Nur Zeitreihenanalyse testen

Testet die Analyse-Komponente isoliert mit simulierten Daten:

```bash
python3 test_with_real_image.py \
    --test-timeseries \
    --num-pairs 10 \
    --db-user aiuser \
    --db-password "DHBW1234!?"
```

**Erwartetes Ergebnis:**
```
================================================================================
📊 ZEITREIHENANALYSE TEST
================================================================================
  Test-Paare: 5
================================================================================

✓ Datenbank verbunden

📝 Erzeuge 5 Test-Paare...
  Paar 1: X=3 (conf=0.85), Y=3 (conf=0.82) ✓
  Paar 2: X=5 (conf=0.78), Y=4 (conf=0.81) ✓
  Paar 3: X=2 (conf=0.65), Y=4 (conf=0.88) ✓
  Paar 4: X=7 (conf=0.92), Y=7 (conf=0.90) ✓
  Paar 5: X=1 (conf=0.55), Y=2 (conf=0.60) ✓

✓ 5 Test-Paare in Datenbank gespeichert

🔍 Führe Zeitreihenanalyse durch...

[Analyse #1] 2024-01-15 10:30:45
--------------------------------------------------------------------------------
✓ 5 Paare gefunden

📈 Analysierte Ergebnisse aus Datenbank:

================================================================================
   X |    Y | Geschätzt | Konfidenz | Zeitdiff | Match | Diff
--------------------------------------------------------------------------------
   3 |    3 |          3 |     0.835 |    0.200s |     ✓ |    0
   5 |    4 |          5 |     0.795 |    0.200s |     ✗ |    1
   2 |    4 |          4 |     0.765 |    0.200s |     ✗ |    2
   7 |    7 |          7 |     0.910 |    0.200s |     ✓ |    0
   1 |    2 |          2 |     0.575 |    0.200s |     ✗ |    1
================================================================================

📊 STATISTIK:
  Übereinstimmungen: 2/5 (40.0%)
  Durchschn. Konfidenz: 0.776
  Durchschn. Zeitdifferenz: 0.200s
================================================================================

✓ Zeitreihenanalyse-Test bestanden
```

---

### Szenario 4: KOMPLETTER PIPELINE-TEST (EMPFOHLEN)

Testet die vollständige Pipeline: Detection → Datenbank → Zeitreihenanalyse

```bash
python3 test_with_real_image.py \
    --sample crowd \
    --test-full-pipeline \
    --db-user aiuser \
    --db-password "DHBW1234!?"
```

**Was wird getestet:**
1. ✅ Personenerkennung mit echtem Foto
2. ✅ Speicherung der Detection in Datenbank
3. ✅ Simulation einer zweiten Detection (leicht variiert)
4. ✅ Zeitreihenanalyse der beiden Detections
5. ✅ Speicherung des korrelierten Ergebnisses

**Erwartetes Ergebnis:**
```
================================================================================
🔄 KOMPLETTER PIPELINE-TEST
================================================================================

SCHRITT 1: Personenerkennung
--------------------------------------------------------------------------------
  Erkannte Personen: 5
  Durchschnittliche Konfidenz: 0.847
  Maximale Konfidenz: 0.923
  Minimale Konfidenz: 0.721


SCHRITT 2: Datenbank-Operationen
--------------------------------------------------------------------------------
✓ Detection X gespeichert (ID: 42)
✓ Detection Y gespeichert (ID: 43)


SCHRITT 3: Zeitreihenanalyse
--------------------------------------------------------------------------------
[Analyse #1] 2024-01-15 10:32:15
--------------------------------------------------------------------------------
✓ 1 Paare gefunden

✓ Korreliertes Ergebnis:
  Geschätzte Personen: 5
  Konfidenz: 0.841

================================================================================
✓ KOMPLETTER PIPELINE-TEST ERFOLGREICH
================================================================================
```

---

## 🔍 Verfügbare Beispielbilder

```bash
python3 test_with_real_image.py --list-samples
```

Ausgabe:
- `crowd` - Menschenmenge (mehrere Personen)
- `person` - Einzelne Person
- `people` - Kleine Gruppe

---

## 🎯 Empfohlene Test-Reihenfolge

### Für Ersteinrichtung:

```bash
# 1. Basis-Check
python3 test_system.py full-test --db-user aiuser --db-password "DHBW1234!?"

# 2. Detection mit echtem Bild
python3 test_with_real_image.py --sample crowd --save-result

# 3. Vollständiger Pipeline-Test
python3 test_with_real_image.py --sample crowd --test-full-pipeline

# 4. System starten
./start_system.sh start

# 5. Mit echten Bildern testen
cp temp_test_images/crowd.jpg input_x/test1.jpg
```

### Für Entwicklung/Debugging:

```bash
# Nur Zeitreihenanalyse (schnell)
python3 test_with_real_image.py --test-timeseries --num-pairs 10

# Nur Detection (ohne Analyse)
python3 test_with_real_image.py --sample person

# Mit niedriger Konfidenz (mehr Detections)
python3 test_with_real_image.py --sample crowd --confidence 0.3

# Pipeline mit eigenem Bild
python3 test_with_real_image.py --image mein_foto.jpg --test-full-pipeline
```

---

## 📊 Test-Ausgaben verstehen

### Detection-Ergebnisse

- **Erkannte Personen**: Anzahl der gefundenen Personen
- **Konfidenz**: Wie sicher ist YOLO (0.0-1.0)?
  - `> 0.8`: Sehr sicher
  - `0.5-0.8`: Mittlere Sicherheit (Standard-Schwelle: 0.5)
  - `< 0.5`: Unsicher (wird nicht angezeigt bei Standard-Einstellung)

### Zeitreihenanalyse-Ergebnisse

- **X/Y**: Rohdaten aus beiden Quellen
- **Geschätzt**: Berechnete tatsächliche Personenzahl
- **Konfidenz**: Vertrauenswürdigkeit der Schätzung
- **Zeitdiff**: Zeit zwischen beiden Detections (sollte < 5s sein)
- **Match**: ✓ = Beide Quellen stimmen überein, ✗ = Abweichung
- **Diff**: Differenz zwischen X und Y

### Analyse-Strategien

Das System verwendet 4 Strategien zur Schätzung:

1. **Übereinstimmung** (Match ✓): Beide Quellen zeigen gleiche Anzahl → direkt übernehmen
2. **Große Abweichung** (Diff > 2): Konservativ → Maximum nehmen
3. **Konfidenz-Differenz** (> 0.1): Höhere Konfidenz gewinnt
4. **Gewichteter Durchschnitt**: Bei ähnlicher Konfidenz

---

## 🐛 Troubleshooting

### "Keine Personen erkannt" bei echten Fotos

```bash
# Versuche niedrigere Konfidenz-Schwelle
python3 test_with_real_image.py --sample crowd --confidence 0.3

# Prüfe ob das Bild wirklich Personen enthält
# (nicht zu klein, nicht zu unscharf, frontal sichtbar)
```

### Datenbank-Fehler

```bash
# Prüfe Verbindung
python3 test_system.py test-db --db-user aiuser --db-password "DHBW1234!?"

# Prüfe ob PostgreSQL läuft
systemctl status postgresql
```

### "No module named 'ultralytics'"

```bash
pip install -r requirements.txt
```

### SSL-Fehler beim Download

```bash
# Verwende eigenes Bild statt Download
python3 test_with_real_image.py --image /pfad/zu/bild.jpg
```

---

## 📈 Performance-Metriken

### Gute Werte:
- **Detection-Zeit**: < 1s pro Bild
- **Konfidenz**: > 0.7 im Durchschnitt
- **Zeitreihen-Übereinstimmung**: > 60%
- **Korrelations-Konfidenz**: > 0.7

### Optimierungen:
- GPU-Beschleunigung aktivieren (automatisch wenn CUDA verfügbar)
- Größeres YOLO-Modell für bessere Genauigkeit: `yolov8m.pt` oder `yolov8l.pt`
- Konfidenz-Schwelle anpassen je nach Anforderung

---

## 🎓 Weiterführende Tests

### Test mit Video-Frames

```python
# Extrahiere Frames aus Video und teste
import cv2

cap = cv2.VideoCapture('video.mp4')
frame_count = 0

while frame_count < 10:
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f'input_x/frame_{frame_count}.jpg', frame)
        frame_count += 1
    else:
        break

cap.release()
```

### Stress-Test

```bash
# Simuliere viele Bilder
python3 test_system.py simulate --duration 300 --interval 0.5
```

### Genauigkeits-Test

```bash
# Teste mit verschiedenen Konfidenzen
for conf in 0.3 0.5 0.7 0.9; do
    echo "Testing with confidence: $conf"
    python3 test_with_real_image.py --sample crowd --confidence $conf
done
```