# Technische Summary des Prognosemodells

## Ziel des Systems

Das Prognosemodell dient dazu, die Auslastung von Bibliotheks- und Lernraeumen der DHBW Mannheim auf Basis historischer Projektdaten und externer Kontextdaten vorherzusagen. Ziel ist nicht nur eine Punktprognose, sondern eine robuste, produktionsnahe Vorhersage mit Unsicherheitsabschaetzung und nachvollziehbarer Explainability.

---

## Datenbasis

Die Modellpipeline basiert auf zwei Hauptquellen:

1. **Historische Projektdaten** aus `KI_Projekt_Daten_einJahr.xlsx` (13.187 Zeilen, 15-Minuten-Intervalle, Jan-Dez 2025)
2. **Externe Kontextdaten** mit DHBW-Bezug, insbesondere Vorlesungs- und Klausurinformationen aus dem DHBW-Vorlesungsplan

Die Daten werden in einer vorbereitenden Pipeline vereinheitlicht, zeitlich ausgerichtet und in ein trainingsfaehiges Format ueberfuehrt.

---

## Implementierte Pipeline

Die Loesung besteht aus fuenf zentralen Bausteinen:

### 1. Datenaufbereitung

**Datei:** `scripts/data/prepare_training_data.py`

- Einlesen der Excel-Daten (13.475 Zeilen)
- Integration externer Lecture-/DHBW-Profile aus SQLite
- Vereinheitlichung der Zeitachsen auf 15-Minuten-Intervalle
- Ausgabe als `training_data.parquet` (13.187 Zeilen, 42 Spalten)

Die Datenaufbereitung trennt Rohdatenlogik von Trainingslogik. Dadurch wird die Pipeline reproduzierbar und wartbar.

### 2. Feature Engineering

**Datei:** `services/forecast/features_excel.py`

- 40 strukturierte Modellmerkmale in 9 Kategorien
- Versionierung ueber `excel_v1`
- Kategorien: Excel-Faktoren (7), Binaer (5), Auslastung (1), Zeitlich/Zyklisch (8), Occupancy-Lags (8), Rolling Stats (5), Diffs (2), Vorlesungs-Proxies (4)

Die Feature-Schicht kapselt die fachliche Vorverarbeitung und reduziert harte Abhaengigkeiten zwischen Rohdaten und Modellcode.

### 3. Modellierung

**Datei:** `services/forecast/model_gbdt.py`

- LightGBM Gradient Boosted Decision Tree
- Quantile Regression mit drei separaten Modellen:
  - **q03** (alpha=0.03) -- untere Grenze des 94%-Prediction-Intervals
  - **q50** (alpha=0.50) -- Median-Prognose
  - **q97** (alpha=0.97) -- obere Grenze des 94%-Prediction-Intervals
- Monotonicity Enforcement: q03 <= q50 <= q97

Das Modell liefert nicht nur einen erwarteten Wert, sondern auch ein Prognoseintervall. Das erhoeht die Nutzbarkeit fuer operative Entscheidungen, weil Unsicherheit explizit sichtbar wird.

### 4. Training und Validierung

**Datei:** `services/forecast/train_gbdt.py`

- 6-Fold Walk-Forward Cross-Validation mit expandierendem Trainingsfenster
- Zeitstempelbasierte Splits (nicht indexbasiert) -- verhindert Data Leakage bei Datenluecken
- 1-Tag-Gap zwischen Training und Test
- Promotion Gate fuer Modellfreigaben (Improvement >= 8%, Coverage in [85%, 98%])
- Gewichtete Metrik-Aggregation nach Test-Set-Groesse

Durch Walk-Forward-Validierung wird die reale Vorhersagesituation besser simuliert als bei zufaelligen Splits. Das ist besonders wichtig bei zeitbezogenen Daten.

### 5. Explainability

**Dateien:** `services/xai/shap_explainer.py`, `services/xai/feature_labels.py`

- SHAP TreeExplainer fuer exakte Shapley-Werte (keine Approximation)
- Lokale Erklaerungen: Top-N Einflussfaktoren pro Einzelprognose
- Globale Erklaerungen: Feature-Importance-Ranking ueber alle Daten
- Deutsche Bezeichnungen fuer alle 40 Features

Das Modell ist dadurch nicht nur leistungsfaehig, sondern auch interpretierbar. Fachanwender koennen nachvollziehen, welche Merkmale eine konkrete Prognose beeinflusst haben.

---

## Anpassungen an der bestehenden Architektur

| Datei | Aenderung |
|-------|-----------|
| `services/forecast/model_tf.py` | Ergaenzung von `build_mlp_v2()` als Referenz zum alten TF-Ansatz |
| `services/forecast/model_store.py` | Unterstuetzung fuer GBDT-Bundles (Laden, Statusabfrage) |
| `services/forecast/main.py` | Neuer Inferenzpfad `_forecast_lgbm()`, Umschaltung ueber `FORECAST_MODEL_BACKEND=lgbm` |
| `services/forecast/requirements.txt` | `lightgbm>=4.0.0`, `shap>=0.43.0`, `pyarrow>=14.0.0` ergaenzt |
| `CLAUDE.md` | Aktualisierung des Projektkontexts und Modellstatus |

---

## Warum ein Wechsel vom alten TF-MLP zum neuen GBDT sinnvoll war

Das vorherige Modell basierte auf einem TensorFlow-MLP. Der neue Ansatz mit LightGBM ist fuer die vorliegenden Daten technisch besser geeignet, weil:

- **Tabellarische Daten mit gemischten Einflussgroessen** typischerweise sehr gut von GBDT-Modellen verarbeitet werden
- **Weniger aufwendiges Skalierungs- und Tuningverhalten** erforderlich ist
- **Nichtlineare Zusammenhaenge und Interaktionen** gut erfasst werden
- **Quantilprognosen direkt modelliert** werden koennen
- **Explainability mit SHAP** fuer Tree-Modelle exakt (nicht approximiert) berechnet wird

---

## Wissenschaftliche Evaluation -- Ergebnisse

Die wissenschaftliche Evaluation wurde am 28.03.2026 abgeschlossen. Das vollstaendige Protokoll liegt unter `docs/scientific-evaluation-report.md`.

### Evaluationsmethodik

- **6-Fold Walk-Forward Cross-Validation** mit zeitstempelbasierten Splits
- **5.997 Out-of-Sample-Datenpunkte** (Jul-Dez 2025)
- **5 faire Baselines** (Persistence H60, Seasonal, Rolling Mean 1h/4h, Global Mean)
- **Statistische Tests:** Diebold-Mariano, Bootstrap-Konfidenzintervalle (2.000 Resamples)
- **Segmentanalyse** nach Tageszeit, Wochentag, Monat und Auslastungsniveau

### Kernkennzahlen

| Metrik | Wert |
|--------|------|
| **MAE GBDT** | **0.798** (95% CI: [0.782, 0.815]) |
| RMSE | 1.035 |
| MdAE (Median) | 0.654 |
| Coverage (94%-PI) | 87.3% |
| Mittlere Intervallbreite | 3.59 |
| Fold-Stabilitaet (CV) | 7.5% |
| Bias (mittlerer Residual) | -0.003 |

### Multi-Baseline-Vergleich

| Baseline | MAE | Improvement | MASE | DM p-Wert | Signifikant |
|----------|-----|-------------|------|-----------|-------------|
| Persistence H60 | 5.619 | +85.8% | 0.142 | < 0.0001 | Ja |
| Seasonal (gestern) | 4.548 | +82.5% | 0.175 | < 0.0001 | Ja |
| Rolling Mean 1h | 4.064 | +80.4% | 0.196 | < 0.0001 | Ja |
| Rolling Mean 4h | 6.539 | +87.8% | 0.122 | < 0.0001 | Ja |
| Global Mean | 5.920 | +86.5% | 0.135 | < 0.0001 | Ja |

**MASE < 1 gegen alle Baselines** -- das Modell ist besser als jede Naive-Methode.

### Vergleich mit altem TF-MLP

| Modell | MAE | Einordnung |
|--------|-----|------------|
| Altes TF-MLP | 10.60 | Schlechter als jede Baseline |
| Persistence H60 (faire Baseline) | 5.619 | Referenz |
| **Neues GBDT** | **0.798** | **Wissenschaftlich validiert, GO-Status** |

### Segmentanalyse -- keine systematischen Schwaechen

| Segment | Bereich | MAE | Bewertung |
|---------|---------|-----|-----------|
| Tageszeit | 10-22h | 0.51-0.94 | Kein Stundensegment ueber 1.0 |
| Wochentag | Mo-Sa | 0.76-0.89 | Mittwoch leicht erhoeht (0.89), kein Ausreisser |
| Auslastung niedrig | [0, 15) | 0.67-0.77 | Sehr gut |
| Auslastung mittel | [15, 30) | 0.87 | Gut |
| Auslastung hoch | [30, 50) | 1.26 | Akzeptabel, aber wenig Daten (n=49) |

### Residualanalyse

- **Bias: -0.003** -- praktisch keine systematische Ueber- oder Unterschaetzung
- **90% der Fehler innerhalb +/-1.7** Personen
- **Schiefe: +0.356** -- leicht rechtsschief, keine problematische Asymmetrie

---

## GO/NO-GO Checkliste

| # | Kriterium | Status | Wert |
|---|-----------|--------|------|
| 1 | MAE besser als alle Baselines | PASS | 0.798 vs 4.064-6.539 |
| 2 | Improvement >= 8% vs Persistence | PASS | 85.8% |
| 3 | Coverage94 in [85%, 98%] | PASS | 87.3% |
| 4 | DM-Test signifikant (p < 0.05) | PASS | p < 0.0001 |
| 5 | Bootstrap-CI schliesst 0 Verbesserung aus | PASS | CI: [4.730, 4.904] |
| 6 | Fold-Stabilitaet CV < 15% | PASS | 7.5% |
| 7 | Bias < 0.5 | PASS | -0.003 |
| 8 | Kein Extremsegment (MAE < 5.0) | PASS | Max: 1.261 |

**Gesamtbewertung: GO** -- alle 8 Kriterien bestanden.

### Technische Validierung (Dry Run)

Zusaetzlich wurde ein Dry Run mit 40 technischen Tests durchgefuehrt (`scripts/evaluation/dry_run.py`):

- Daten-Reproduzierbarkeit, Feature-Determinismus, Zeitstempel-Ordnung
- Walk-Forward-Splits: disjunkt, chronologisch, expandierend
- Model-Roundtrip: Laden, Predict, Determinismus
- Quantil-Konsistenz: q03 <= q50 <= q97 fuer alle 13.183 Samples
- Feature-Alignment Training vs Inferenz
- Feature-Labels vollstaendig
- Error Handling fuer Randfaelle
- Metriken-Plausibilitaet

**Ergebnis: 40/40 Tests bestanden.**

---

## Bekannte Einschraenkungen

### 1. Feature-Mismatch Training vs Live-Inferenz

Das Training nutzt `features_excel.py` (15-Min-Intervalle), die Live-Inferenz nutzt `features.py` (1-Min-Intervalle). Ein Mapping-Layer (`LIVE_TO_EXCEL` in `main.py`) ueberbrueckt die Differenz:

- 26 von 40 Features werden direkt oder ueber Mapping bereitgestellt
- 7 Excel-spezifische Faktoren (f_month, f_weekday, f_tod etc.) werden aus Zeitstempeln abgeleitet
- 7 weitere Features (occupancy_lag_day, occupancy_lag_week, bridge_day, winter_break, weather_rainy, weather_sunny, is_partial_closure) werden mit 0 gefuellt

**Empfehlung:** Mittelfristig das Modell mit `features.py`-Features retrainieren.

### 2. Quantil-Kalibrierung

Die Coverage (87.3%) liegt leicht unter dem theoretischen Ziel von 94%. Die Prediction Intervals sind etwas zu eng. Fuer praktische Zwecke akzeptabel (>= 85%), aber eine Nachkalibrierung (z.B. Conformal Prediction) koennte die Zuverlaessigkeit erhoehen.

### 3. Datenbasis

- **1 Jahr Daten** -- keine jahresuebergreifende Validierung moeglich
- **Wenig High-Load-Daten** -- nur 49 von 5.997 OOS-Samples mit Belegung > 30
- **Keine Sonntage** in den Daten

### 4. Explainability-Grenzen

SHAP erklaert den Beitrag von Merkmalen zur Modellentscheidung, nicht die reale Ursache im wissenschaftlichen Sinn. Korrelation ist nicht gleich Kausalitaet. Zudem sind Lag-, Rolling-Mean- und Diff-Features hochkorreliert -- SHAP verteilt Importance zwischen ihnen, einzelne Werte koennen instabil sein.

### 5. Semantik von occupancy_lag_day und occupancy_lag_week

Die Feature-Namen suggerieren "gestern gleiche Uhrzeit" bzw. "letzte Woche gleiche Uhrzeit". Tatsaechlich verwendet der Code:

- `occupancy_lag_day`: `shift(48)` = 48 × 15 min = **12 Stunden** (ein Betriebstag = Oeffnungszeit ~8-20 Uhr)
- `occupancy_lag_week`: `shift(288)` = 288 × 15 min = **72 Stunden** (6 Werktage × 12h Betriebsstunden)

Das ist fachlich korrekt -- ein "Betriebstag" der Bibliothek umfasst ca. 12 Stunden Oeffnungszeit, nicht 24 Stunden. Die zeitlichen Abstaende sind also als Abstaende in Betriebsstunden zu interpretieren, nicht in Kalenderzeit. **Datei:** `scripts/data/prepare_training_data.py:173,178`

---

## Kapazitaetsgrenzen und Bug-Fix (April 2026)

### Problembeschreibung

Bis April 2026 lieferten alle vier Live-Inferenzpfade Prognosewerte, die die physikalische Raumkapazitaet ueberschreiten konnten. Das Modell selbst ist korrekt trainiert (Target wird in Training auf `[0, 84]` geklemmt), die Live-Inferenzpfade klemmten Ausgaben jedoch nur nach unten (`max(0.0, ...)`), nicht nach oben.

**Symptom:** Langfristige Prognosen (insbesondere `_forecast_baseline` mit linearer Extrapolation) konnten Werte weit ueber 84 produzieren.

### Technische Ursache

Die Funktion `predict_gbdt()` (`services/forecast/model_gbdt.py`) klemmte korrekt auf `[0, capacity=84]` -- wurde aber **ausschliesslich** in `train_gbdt.py` waehrend der Fold-Evaluation aufgerufen. Im Live-Inferenzpfad (`main.py`) wurde `bundle.model_q50.predict(X_inf)` ohne obere Begrenzung aufgerufen.

Das Baseline-Modell (LinearRegression auf Trend + Saisonalitaet) extrapoliert linear -- ohne obere Grenze produziert es bei laengeren Horizonten arbitraer hohe Werte.

### Betroffene Prognose-Pfade

| Pfad | Datei | Funktion | Status vor Fix |
|------|-------|----------|----------------|
| LGBM Inferenz | `services/forecast/main.py` | `_forecast_lgbm()` | Nur `max(0.0, ...)`, kein Capacity-Clip |
| TF Inferenz | `services/forecast/main.py` | `_forecast_tf()` | Nur `max(0.0, ...)`, kein Capacity-Clip |
| Baseline-Extrapolation | `services/forecast/main.py` | `_forecast_baseline()` | Linear unbegrenzt |
| Wochen-Prognose | `services/forecast/weekly.py` | `build_weekly_forecast()` | Kein Clipping |
| Baseline in Eval | `services/forecast/scientific_eval.py` | `_baseline_point_forecast()` | Linear unbegrenzt (Metrik-Artefakt) |

### Fix-Implementierung

**Neue Hilfsfunktion** `_get_zone_capacity(zone_id)` in `main.py` liest Kapazitaet zonenspezifisch aus der Datenbank:

```python
def _get_zone_capacity(zone_id: str) -> float:
    with SessionLocal() as db:
        row = db.execute(
            text("SELECT capacity FROM zones WHERE zone_id = :zone_id LIMIT 1"),
            {"zone_id": zone_id},
        ).first()
    return float(row._mapping["capacity"]) if row is not None else float(os.getenv("DEFAULT_ZONE_CAPACITY", "100"))
```

Alle Inferenzpfade cllipen jetzt auf `[0, capacity]`. Im LGBM-Pfad wird zusaetzlich ein Warning-Log ausgegeben, wenn der Rohwert die Kapazitaet ueberschreitet -- fruehzeitiger Indikator fuer Feature-Drift oder Retraining-Bedarf:

```python
if q50_pred > capacity or q97_pred > capacity:
    logger.warning(f"LGBM raw output exceeds capacity={capacity:.0f}: q50={q50_pred:.1f}, q97={q97_pred:.1f} ...")
q03_pred = max(0.0, min(capacity, q03_pred))
q50_pred = max(0.0, min(capacity, q50_pred))
q97_pred = max(0.0, min(capacity, q97_pred))
q03_pred = min(q03_pred, q50_pred)   # Monotonicity
q97_pred = max(q97_pred, q50_pred)   # Monotonicity
```

### Kapazitaetsbestimmung

Die Kapazitaet fuer `default-zone` betraegt 84 Personen. Dieser Wert ist definiert in:

- `services/forecast/features_excel.py:CAPACITY_TOTAL = 84` -- Training/Feature Engineering
- `scripts/data/prepare_training_data.py:CAPACITY_TOTAL = 84` -- Datenaufbereitung
- `infra/db/migrations/001_init.sql` -- DB-Spalte `zones.capacity`

**Prioritaetsreihenfolge im Inference-Pfad:** DB-Wert (`zones.capacity`) > `.env` (`DEFAULT_ZONE_CAPACITY=100`) > Hardcoded Fallback (100).

### Verifikation

| Datei | Zweck |
|-------|-------|
| `scripts/tests/test_forecast_capacity_bug.py` | Reproduziert die 3 urspruenglichen Bugs (erwarteter Exit 1) |
| `scripts/tests/test_forecast_capacity_fix.py` | Validiert alle 4 Inferenzpfade nach Fix -- alle 4 Tests PASS |

Testabdeckung: Baseline-Clipping, LGBM-Clipping, Weekly-Clipping, Edge Cases (Null, Negativ, Grenzwert, Monotonicity).

### Verbleibende Risiken

| Bereich | Status | Begruendung |
|---------|--------|-------------|
| `scientific_eval.py` Z. 843, 962 (TF-Eval) | Nicht gefixt | Eval-Only-Pfad, TF ist nicht primaeres Modell, kein Produktionseinfluss |
| Dashboard-UI | Nicht notwendig | API clippt korrekt, keine doppelte Begrenzung noetig |

---

## Staerken des neuen Setups

1. **Klare Trennung der Verantwortlichkeiten** zwischen Datenaufbereitung, Feature Engineering, Training, Inferenz und Explainability
2. **Zeitlich sinnvolle Validierung** durch Walk-Forward mit zeitstempelbasierten Splits statt zufaelliger Splits
3. **Geeigneter Modelltyp** fuer strukturierte/tabellarische Daten
4. **Unsicherheitsabschaetzung** durch Quantile statt nur einer Punktprognose
5. **Nachvollziehbarkeit** durch SHAP und deutsche Feature-Labels
6. **Produktionsnaehe** durch Bundle-Support und Backend-Umschaltung
7. **Wissenschaftliche Absicherung** durch Multi-Baseline-Vergleich, statistische Tests und dokumentierte Evaluation

---

## Antworten auf wahrscheinliche Professorenfragen

### "Warum ist das besser als vorher?"

Weil das neue System nicht nur ein anderes Modell verwendet, sondern die gesamte Pipeline verbessert wurde. Die Rohdatenaufbereitung wurde strukturiert, das Feature Engineering versioniert, die Validierung zeitlich sauberer gemacht, Unsicherheiten werden ueber Quantile modelliert und die Vorhersagen sind ueber SHAP nachvollziehbar. Dadurch ist das System nicht nur genauer, sondern auch robuster und besser erklaerbar.

### "Warum LightGBM und nicht ein neuronales Netz?"

Fuer tabellarische Daten mit begrenzter historischer Tiefe und heterogenen Einflussgroessen sind GBDT-Modelle wie LightGBM in der Praxis oft staerker, stabiler und einfacher interpretierbar als ein MLP. Der Wechsel war daher nicht nur empirisch, sondern auch architektonisch plausibel. Konkret: Das alte TF-MLP erreichte eine MAE von 10.60, das neue GBDT 0.798 -- bei identischer Datenbasis.

### "Ist das wissenschaftlich bewiesen?"

Ja. Das Modell wurde in einer 6-Fold Walk-Forward Cross-Validation mit 5.997 Out-of-Sample-Datenpunkten evaluiert. Es schlaegt alle 5 fairen Baselines statistisch hochsignifikant (Diebold-Mariano, p < 0.0001). Die MAE liegt bei 0.798 mit einem 95%-Bootstrap-Konfidenzintervall von [0.782, 0.815]. Die Segmentanalyse zeigt keine systematischen Schwaechen. Der vollstaendige Evaluationsbericht liegt unter `docs/scientific-evaluation-report.md`.

### "Wie robust ist das Modell ueber verschiedene Zeitraeume?"

Die 6 Walk-Forward-Folds decken Jul-Dez 2025 ab. Die Fold-MAEs liegen zwischen 0.749 und 0.924 mit einem Variationskoeffizienten von 7.5%. Das zeigt eine sehr stabile Performance ohne saisonale Einbrueche.

### "Was passiert bei hoher Auslastung?"

Bei Belegungen ueber 30 steigt die MAE auf 1.26. Allerdings gibt es nur 49 solcher Datenpunkte (0.8% der OOS-Daten). Fuer eine belastbare Aussage ueber High-Load-Szenarien werden mehr Daten benoetigt.

### "Koennte es Data Leakage geben?"

Drei Massnahmen schliessen Leakage aus: (1) Walk-Forward-Splits mit 1-Tag-Gap zwischen Training und Test, (2) zeitstempelbasierte Boundaries statt indexbasierter Splits, (3) Lag-Features werden korrekt mit `.shift()` (rueckwaerts) berechnet, Targets mit `.shift(-4)` (vorwaerts). Der Dry Run hat die zeitliche Integritaet aller Splits verifiziert (disjunkt, chronologisch, expandierend).

---

## Naechste sinnvolle Schritte

### Kurzfristig (nach Produktivfreigabe)
1. Live-Monitoring der Prognosequalitaet (MAE, Coverage) einrichten
2. Retraining mit `features.py`-Features um Feature-Mismatch zu eliminieren
3. Quantil-Nachkalibrierung (Conformal Prediction) pruefen

### Mittelfristig
1. Jahresuebergreifende Validierung mit Daten aus 2026
2. Online-Learning / regelmaessiges Retraining
3. Segment-spezifische Modelle fuer High-Load-Perioden
