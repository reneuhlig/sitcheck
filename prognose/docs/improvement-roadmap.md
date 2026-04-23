# Verbesserungs-Roadmap: Sitcheck Prognosemodell

**Stand:** 2026-03-28
**Bezug:** [Model Evaluation Report](./model-evaluation-report.md)

---

## A. Quick Wins (je 0.5-2 Tage, hoher Impact)

### QW1: Horizon-Output auf Terminal-Wert reduzieren
**Prioritaet:** KRITISCH

**Was:** Statt 60 gleichzeitiger Outputs (y_tplus_1 bis y_tplus_60) nur den Terminal-Wert (y_tplus_60) oder wenige Key-Horizonte (t+15, t+30, t+60) vorhersagen.

**Warum:** Der aktuelle Multi-Step-Output zwingt das Modell, einen Kompromiss ueber 60 Zeitschritte zu optimieren. Fruehe Steps sind trivial und verwaessern den Gradienten fuer spaete, schwierigere Steps. Die Baseline wird nur am Terminal-Punkt gemessen, was den Vergleich verzerrt.

**Dateien:**
- `services/forecast/model_tf.py:40` -- Output-Layer von `horizon` auf 1 (oder 3) aendern
- `services/forecast/features.py:259-262` -- `build_supervised_dataset()` auf Terminal-Target anpassen
- `services/forecast/train_tf.py:120-141` -- Evaluation auf neues Target-Format anpassen

**Erwarteter Nutzen:** MAE-Reduktion um geschaetzt 30-50%, fairer Baseline-Vergleich
**Aufwand:** 1 Tag

---

### QW2: Feature-Scaling nach Typ trennen
**Prioritaet:** HOCH

**Was:** StandardScaler nur auf numerische Features (Lags, Rolling Stats, Diffs) anwenden. Sin/Cos-Features und binaere Features unverarbeitet lassen.

**Warum:** `StandardScaler` auf bereits normierte Features (`minute_sin` in [-1,1], `event_active` in {0,1}) zerstoert deren semantische Bedeutung und verschlechtert die Modellleistung.

**Dateien:**
- `services/forecast/train_tf.py:88-91` -- ColumnTransformer statt globalem StandardScaler

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

PASSTHROUGH_FEATURES = [
    "minute_sin", "minute_cos", "dow_sin", "dow_cos",
    "event_active", "lecture_low_period_flag",
]

numeric_features = [f for f in feature_columns if f not in PASSTHROUGH_FEATURES]

scaler = ColumnTransformer([
    ("scale", StandardScaler(), numeric_features),
    ("pass", "passthrough", PASSTHROUGH_FEATURES),
])
```

**Erwarteter Nutzen:** Verbesserte Feature-Nutzung, besonders fuer zyklische und binaere Features
**Aufwand:** 0.5 Tage

---

### QW3: Evaluations-Paritaet sicherstellen
**Prioritaet:** KRITISCH

**Was:** Sicherstellen, dass TF-Modell und Baseline auf exakt denselben Forecast-Punkten verglichen werden -- entweder beide auf dem Terminal-Punkt oder beide ueber alle Steps.

**Warum:** Die aktuelle Baseline (`scientific_eval.py:242`) gibt nur `yhat = yhat_path[-1]` zurueck (Terminal-Wert), waehrend das TF-Modell ueber alle 60 Steps evaluiert wird. Das macht den 7.42x-Gap teilweise zu einem Artefakt.

**Dateien:**
- `services/forecast/scientific_eval.py:192-248` -- Baseline auf denselben Evaluationsmodus bringen
- `services/forecast/train_tf.py:136-141` -- Metriken konsistent berechnen

**Erwarteter Nutzen:** Realistischer Performance-Vergleich; der wahre Gap ist wahrscheinlich deutlich kleiner
**Aufwand:** 1 Tag

---

### QW4: Target-Clipping auf [0, capacity]
**Prioritaet:** HOCH

**Was:** Alle Modellausgaben auf `max(0, prediction)` clippen; optional obere Grenze auf Zonenkapazitaet.

**Warum:** Negative Belegungsprognosen sind physikalisch unmoeglich. Die aktuellen Residual-basierten Prediction Intervals (`train_tf.py:132-134`) koennen `pi_low < 0` erzeugen.

**Dateien:**
- `services/forecast/main.py` -- Post-Processing-Schritt nach Prediction
- `services/forecast/train_tf.py:132-134` -- Residuals-Berechnung anpassen

**Erwarteter Nutzen:** Keine unsinnigen Prognosen, bessere Coverage-Metriken
**Aufwand:** 0.5 Tage

---

### QW5: Gradient Clipping und LR Warmup
**Prioritaet:** MITTEL

**Was:** `clipnorm=1.0` zum Adam-Optimizer hinzufuegen; optional Cosine-Annealing LR-Schedule.

**Warum:** Multi-Output-Modelle mit Huber-Loss koennen instabile Gradienten erzeugen. Gradient Clipping stabilisiert das Training.

**Dateien:**
- `services/forecast/model_tf.py:44` -- `Adam(learning_rate=1e-3, clipnorm=1.0)`

**Erwarteter Nutzen:** Stabileres Training, weniger Varianz zwischen Runs
**Aufwand:** 0.5 Tage

---

## B. Mittelfristige Verbesserungen (je 1-2 Wochen)

### MT1: Temporale Modellarchitektur (1D-CNN oder LSTM)
**Prioritaet:** HOCH

**Was:** Das aktuelle MLP durch eine Architektur ersetzen, die temporale Abhaengigkeiten explizit modellieren kann:
- **Option A:** 1D-CNN (Conv1D + MaxPooling + Dense Head) -- schnell, gut fuer lokale Muster
- **Option B:** LSTM/GRU Encoder-Decoder -- besser fuer langfristige Abhaengigkeiten
- **Option C:** Transformer-basiert (Temporal Fusion Transformer) -- State-of-the-Art, aber komplex

**Warum:** Ein MLP sieht die 37 Features als flachen Vektor. Es kann keine sequenzielle Struktur lernen (z.B. "Belegung steigt seit 30 Minuten"). Ein temporales Modell erhaelt die letzten N Minuten als Sequenz.

**Dateien:**
- `services/forecast/model_tf.py` -- Neues Modell definieren
- `services/forecast/features.py` -- Sequence-Input statt Flat-Vector
- `services/forecast/train_tf.py` -- Training anpassen

**Erwarteter Nutzen:** Signifikante MAE-Verbesserung, besonders fuer laengere Horizonte
**Aufwand:** 1 Woche
**Risiko:** Erhoehte Trainingszeit, GPU empfohlen

---

### MT2: LightGBM/XGBoost als Challenger
**Prioritaet:** HOCH

**Was:** Gradient Boosted Decision Trees als alternatives Modell implementieren. Der Platzhalter `MODEL_GBDT = "quantile_gbdt"` existiert bereits (`scientific_eval.py:31`), aber die Implementation fehlt.

**Warum:** GBDTs uebertreffen neuronale Netze konsistent auf tabellarischen Daten mit <100K Zeilen (vgl. Grinsztajn et al., 2022). Sie benoetigen kein Feature-Scaling, sind robust gegenueber irrelevanten Features und trainieren schnell.

**Dateien:**
- `services/forecast/model_gbdt.py` -- Neues Modul erstellen (LightGBM + Quantile Regression)
- `services/forecast/scientific_eval.py:31` -- `MODEL_GBDT`-Integration vervollstaendigen
- `services/forecast/main.py` -- Backend-Switch erweitern

**Erwarteter Nutzen:** Wahrscheinlich sofortige Verbesserung gegenueber MLP; native Feature Importance
**Aufwand:** 1 Woche

---

### MT3: Walk-Forward-Validation im Training
**Prioritaet:** MITTEL

**Was:** Den starren 70/15/15-Split (`train_tf.py:37-46`) durch eine expandierende oder gleitende Fenster-Validation ersetzen.

**Warum:** Der aktuelle Test-Split liegt am zeitlichen Ende der Daten. Wenn sich die Datenverteilung aendert (z.B. Semesterwechsel), ist der Test-Split nicht repraesentativ. Walk-Forward validiert auf mehreren Zeitabschnitten.

**Dateien:**
- `services/forecast/train_tf.py:37-46` -- Split-Logik ersetzen

**Erwarteter Nutzen:** Reduziert Val-Test-Gap (aktuell 2.26x), robustere Modellselektion
**Aufwand:** 1 Woche

---

### MT4: Feature Selection via Permutation Importance
**Prioritaet:** MITTEL

**Was:** Nach dem Training Permutation Importance auf dem Validierungsset berechnen. Features mit negligible Importance entfernen.

**Warum:** 37 Features koennen Rauschen einfuehren, insbesondere wenn einzelne Features wenig Varianz haben oder stark korreliert sind. Feature Selection verbessert Generalisierung.

**Dateien:**
- `services/forecast/train_tf.py` -- Permutation Importance nach Training berechnen
- `services/forecast/scientific_eval.py` -- Importance in Report aufnehmen
- `services/forecast/features.py` -- Optional: Feature-Pruning-Modus

**Erwarteter Nutzen:** Weniger Overfitting, schnelleres Training, bessere Interpretierbarkeit
**Aufwand:** 1 Woche

---

### MT5: Quantile Regression statt Residual-basierter PI
**Prioritaet:** MITTEL

**Was:** Statt Prediction Intervals aus Validierungs-Residuen abzuleiten, direkt q10, q50, q90 als Modell-Outputs trainieren (Pinball Loss pro Quantil).

**Warum:** Residual-basierte Intervals (`train_tf.py:132-134`) sind statisch und nehmen stationaere Fehlerverteilung an. Quantile Regression liefert punktabhaengige, dynamische Unsicherheitsschaetzungen.

**Dateien:**
- `services/forecast/model_tf.py` -- 3 Output-Koepfe (q10, q50, q90) oder Multi-Task-Loss
- `services/forecast/train_tf.py` -- Pinball Loss implementieren
- `services/forecast/main.py` -- Inference anpassen

**Erwarteter Nutzen:** Besser kalibrierte Unsicherheit, dynamische Coverage
**Aufwand:** 1.5 Wochen

---

## C. Strategische Verbesserungen (je 1+ Monate)

### ST1: Ensemble spezialisierter Modelle
**Prioritaet:** HOCH

**Was:** Separate Modelle fuer verschiedene Horizont-Regime:
- **Kurzfristig (H60):** 1D-CNN oder GBDT mit minutengenauen Features
- **Taeglich (H1440):** GBDT mit Tagesaggregaten
- **Woechentlich (H10080/H20160):** Saisonales Modell mit Wochen-Features

**Warum:** Ein einziges Modell kann nicht gleichzeitig Minuten- und Wochenmuster optimal lernen. Die Modellarchitektur, das Feature-Set und die Trainingstrategie sollten zum jeweiligen Horizont passen.

**Erwarteter Nutzen:** Optimale Performance pro Horizont
**Aufwand:** 1 Monat

---

### ST2: Externe Features (Wetter, Ferien, Pruefungen)
**Prioritaet:** MITTEL

**Was:** Zusaetzliche Feature-Quellen integrieren:
- **Wetter:** Temperatur, Regen (Open-Meteo API, kostenlos)
- **Akademischer Kalender:** Semesterphasen, Pruefungszeitraeume, Ferien
- **Feiertage:** Gesetzliche und regionale Feiertage

**Warum:** Die Bibliotheksbelegung korreliert stark mit Wetter (schlechtes Wetter = mehr Indoor) und akademischem Kalender (Pruefungsphase = hohe Belegung).

**Dateien:**
- `services/forecast/features.py` -- Neue Feature-Builder
- Neuer Ingest-Service oder Erweiterung von `lecture-ingest`

**Erwarteter Nutzen:** Erfassung wichtiger externer Einflussfaktoren
**Aufwand:** 2-3 Wochen

---

### ST3: Online-Learning / Inkrementelle Updates
**Prioritaet:** NIEDRIG

**Was:** Statt naechtlichem Batch-Retraining inkrementelle Modell-Updates bei neuen Datenpunkten.

**Warum:** Die Datenverteilung aendert sich (Semesterwechsel, Events). Online-Learning kann sich schneller anpassen als Nightly-Batch.

**Erwarteter Nutzen:** Schnellere Adaption an Verteilungsaenderungen
**Aufwand:** 1 Monat

---

### ST4: MLflow/DVC fuer Experiment-Tracking
**Prioritaet:** MITTEL

**Was:** Experiment-Tracking-System einfuehren (MLflow oder DVC) fuer:
- Modellversionen und Hyperparameter
- Feature-Set-Versionen
- Trainings-Metriken und -Artefakte
- A/B-Vergleiche zwischen Modellkandidaten

**Warum:** Aktuell werden Trainingslaeufe in JSON-Reports gespeichert (`scientific_eval.py:84-100`). Das ist funktional, aber nicht fuer systematischen Modellvergleich optimiert.

**Erwarteter Nutzen:** Reproduzierbarkeit, systematischer Modellvergleich, Team-Zusammenarbeit
**Aufwand:** 2 Wochen

---

## Priorisierungs-Matrix

```
Impact
  ^
  |  QW1 *** QW3 ***
  |  MT2 **  MT1 **
  |  QW2 **  ST1 **
  |  QW4 *   MT5 *
  |  MT4 *   ST2 *
  |  QW5 *   MT3 *
  |  ST4 *   ST3
  +----------------------------> Aufwand
    0.5d  1d  1w  2w  1m
```

**Empfohlene Reihenfolge:**
1. QW1 + QW3 (parallel, 1-2 Tage) -- beseitigt das Evaluations-Artefakt
2. QW2 + QW4 (parallel, 1 Tag) -- verbessert Trainingsqualitaet
3. MT2 (1 Woche) -- GBDT als schneller Challenger
4. MT1 (1 Woche) -- Temporale Architektur
5. MT4 + MT5 (2 Wochen) -- Feature Selection + Quantile Regression
6. ST1 + ST2 (1-2 Monate) -- Ensemble + externe Features
