# Wissenschaftliche Evaluation -- GBDT Prognosemodell

**Modell:** LightGBM Quantile Regression (q03/q50/q97)
**Feature-Set:** excel_v1, 40 Features
**Horizont:** 60 Minuten (4 x 15-Min-Schritte)
**Datengrundlage:** KI_Projekt_Daten_einJahr.xlsx, 13.187 Zeilen, Jan-Dez 2025
**Evaluationsdatum:** 28.03.2026
**Methodik:** 6-Fold Walk-Forward Cross-Validation (zeitstempelbasiert, expandierendes Fenster)

---

## 1. Zusammenfassung

Das GBDT-Modell erzielt eine **MAE von 0.798** (95% CI: [0.782, 0.815]) bei einem mittleren Belegungsniveau von ~12.9. Dies entspricht einem relativen Fehler von ~6.2%.

Die Verbesserung gegenueber allen 5 Baselines ist **statistisch hochsignifikant** (Diebold-Mariano, p < 0.001) und **praktisch relevant** (Improvement >= 80% gegen jede Baseline).

**Gesamtbewertung: GO** -- alle 8 Kriterien der GO/NO-GO-Checkliste bestanden.

---

## 2. Methodik

### 2.1 Walk-Forward Cross-Validation

- **6 Folds**, expandierendes Trainingsfenster
- **Zeitstempelbasierte Splits** (nicht indexbasiert) -- verhindert Data Leakage bei Datenluecken
- **Gap:** 96 Zeitschritte (= 1 Tag) zwischen Training/Validierung und Test
- **Validation Set:** 50% der Test-Dauer, fuer Early Stopping

| Fold | Trainingsende | Testperiode | Test-Samples |
|------|--------------|-------------|-------------|
| 1 | Jul 2025 | Jul-Aug 2025 | 981 |
| 2 | Aug 2025 | Aug-Sep 2025 | 1.004 |
| 3 | Sep 2025 | Sep-Okt 2025 | 991 |
| 4 | Okt 2025 | Okt-Nov 2025 | 1.007 |
| 5 | Nov 2025 | Nov 2025 | 939 |
| 6 | Nov 2025 | Nov-Dez 2025 | 1.075 |

**Gesamt Out-of-Sample:** 5.997 Datenpunkte

### 2.2 Baselines

| Baseline | Beschreibung | Fairness |
|----------|-------------|----------|
| Persistence H60 | Belegung vor 60 Min als Vorhersage | Fairer Horizont-Match |
| Seasonal (gestern) | Gleiche Uhrzeit am Vortag | Standard-Benchmark |
| Rolling Mean 1h | Gleitender Mittelwert 1 Stunde | Einfacher Glaetter |
| Rolling Mean 4h | Gleitender Mittelwert 4 Stunden | Langfristiger Glaetter |
| Global Mean | Mittelwert aller Trainingsdaten | Untergrenze |

---

## 3. Ergebnisse

### 3.1 Punktprognose-Metriken

| Metrik | GBDT | Persistence H60 | Seasonal | Rolling 1h | Rolling 4h | Global Mean |
|--------|------|-----------------|----------|-----------|-----------|-------------|
| **MAE** | **0.798** | 5.619 | 4.548 | 4.064 | 6.539 | 5.920 |
| RMSE | 1.035 | -- | -- | -- | -- | -- |
| MdAE | 0.654 | -- | -- | -- | -- | -- |

### 3.2 Verbesserung gegenueber Baselines

| Baseline | Improvement | MASE | DM-Statistik | p-Wert | Signifikant |
|----------|------------|------|-------------|--------|-------------|
| Persistence H60 | **+85.8%** | 0.142 | -32.96 | < 0.0001 | Ja |
| Seasonal | +82.5% | 0.175 | -14.84 | < 0.0001 | Ja |
| Rolling Mean 1h | +80.4% | 0.196 | -32.07 | < 0.0001 | Ja |
| Rolling Mean 4h | +87.8% | 0.122 | -28.96 | < 0.0001 | Ja |
| Global Mean | +86.5% | 0.135 | -26.16 | < 0.0001 | Ja |

**MASE < 1 gegen alle Baselines** -- das Modell ist besser als jede Naive-Methode.

### 3.3 Bootstrap-Konfidenzintervalle

- **MAE = 0.798**, 95% CI: [0.782, 0.815]
- **MAE-Differenz vs Persistence = 4.821**, 95% CI: [4.730, 4.904]
- Das Konfidenzintervall der Verbesserung schliesst 0 nicht ein -- **statistisch gesichert.**

### 3.4 Quantile und Prediction Intervals

| Metrik | Wert | Ziel |
|--------|------|------|
| Coverage (94%-PI) | **87.3%** | [85%, 98%] -- OK |
| Pinball q03 | 0.077 | -- |
| Pinball q50 | 0.399 | -- |
| Pinball q97 | 0.093 | -- |
| Mittlere Intervallbreite | 3.59 | -- |

**Hinweis:** Das Modell trainiert mit alpha=0.03/0.97, was einem 94%-Prediction-Interval entspricht. Die Coverage von 87.3% liegt leicht unter dem theoretischen 94%, was auf leicht zu enge Intervalle hindeutet. Fuer praktische Zwecke ist dies aber akzeptabel (>= 85%).

---

## 4. Segment-Analyse

### 4.1 Nach Tageszeit

| Stunde | MAE | n | Bewertung |
|--------|-----|---|-----------|
| 10-14h | 0.79-0.86 | je ~516 | Stabil |
| 15-17h | 0.86-0.94 | je ~516 | Leicht erhoehter Fehler (Nachmittags-Peak) |
| 18-21h | 0.51-0.82 | 433-456 | Gut bis sehr gut |

**Kein Stundensegment ueber MAE 1.0** -- keine systematische Schwaeche.

### 4.2 Nach Wochentag

| Tag | MAE |
|-----|-----|
| Mo | 0.757 |
| Di | 0.783 |
| Mi | 0.891 |
| Do | 0.780 |
| Fr | 0.774 |
| Sa | 0.803 |

**Mittwoch leicht erhoehter Fehler (0.891)**, aber kein Ausreisser.

### 4.3 Nach Auslastungsniveau

| Niveau | Bereich | MAE | n |
|--------|---------|-----|---|
| Sehr niedrig | [0, 5) | 0.672 | 813 |
| Niedrig | [5, 15) | 0.771 | 2.840 |
| Mittel | [15, 30) | 0.866 | 2.295 |
| Hoch | [30, 50) | 1.261 | 49 |

**Erwartbar:** Hoehere Auslastung = hoeherer Fehler. Aber nur 49 Samples in "hoch" -- geringe statistische Belastbarkeit fuer dieses Segment.

---

## 5. Residual-Analyse

| Kennzahl | Wert | Bewertung |
|----------|------|-----------|
| Mittlerer Residual (Bias) | -0.003 | Praktisch unbias |
| Std Residual | 1.035 | -- |
| Schiefe | +0.356 | Leicht rechtsschief (mehr Unterschaetzung) |
| Kurtosis | 2.266 | Leicht leptokurtisch |
| Max Ueberschaetzung | -4.7 | Moderat |
| Max Unterschaetzung | +8.1 | Einzelner Ausreisser |
| P5/P95 | [-1.6, 1.7] | 90% der Fehler innerhalb +-1.7 |

**Fazit:** Keine systematische Ueber- oder Unterschaetzung. Die Fehlerverteilung ist nahezu symmetrisch.

---

## 6. Fold-Stabilitaet

| Fold | MAE |
|------|-----|
| 1 | 0.753 |
| 2 | 0.811 |
| 3 | 0.924 |
| 4 | 0.783 |
| 5 | 0.769 |
| 6 | 0.749 |

**Variationskoeffizient: 7.5%** -- sehr stabile Performance ueber alle Zeitperioden.

Fold 3 (Sep-Okt 2025) ist der schwaechste -- moeglicherweise durch Semesterstart und veraenderte Nutzungsmuster.

---

## 7. Explainability-Bewertung

### 7.1 Top-10 Features (LightGBM native Importance)

| Rang | Feature | Importance | Deutsche Bezeichnung |
|------|---------|-----------|---------------------|
| 1 | occupancy_roll_std_8 | 727 | Volatilitaet (2h) |
| 2 | occupancy_roll_mean_16 | 574 | Mittelwert (4h) |
| 3 | f_tod | 530 | Tageszeit-Faktor |
| 4 | minute_cos | 476 | Tageszeit (cos) |
| 5 | occupancy_roll_mean_4 | 473 | Mittelwert (1h) |
| 6 | utilization_pct | 456 | Aktuelle Auslastung |
| 7 | occupancy_roll_std_4 | 419 | Volatilitaet (1h) |
| 8 | f_month | 391 | Monatl. Saisoneffekt |
| 9 | occupancy_diff_4 | 390 | Trend (1h) |
| 10 | occupancy_roll_mean_8 | 375 | Mittelwert (2h) |

### 7.2 Methodische Einschraenkungen

1. **Korrelation != Kausalitaet:** SHAP-Werte zeigen Beitraege zur Vorhersage, nicht kausale Zusammenhaenge.
2. **Feature-Korrelation:** Lags, Rolling-Means und Diffs sind hochkorreliert -- SHAP verteilt Importance zwischen ihnen.
3. **Tree-SHAP ist exakt:** Im Gegensatz zu Kernel-SHAP liefert TreeExplainer exakte Shapley-Werte.

---

## 8. Bekannte Einschraenkungen

### 8.1 Feature-Mismatch Training vs Inferenz

Das Training nutzt `features_excel.py` (15-Min-Intervalle), die Live-Inferenz nutzt `features.py` (1-Min-Intervalle). Ein Mapping-Layer ueberbrueckt die Differenz, aber:

- **14 von 40 Features** werden bei Live-Inferenz entweder approximiert (f_tod, f_month etc.) oder sind nicht verfuegbar (occupancy_lag_day, occupancy_lag_week)
- **7 Excel-spezifische Faktoren** (f_month, f_weekday, f_tod, f_weather, f_bridge, efficiency, capacity_effective) werden aus Zeitstempeln/Defaults abgeleitet
- **2 der Top-10 Features** (f_tod, f_month) sind approximiert -- die Live-Performance koennte leicht schlechter sein

**Empfehlung:** Mittelfristig das Modell mit `features.py`-Features retrainieren, um den Mismatch zu eliminieren.

### 8.2 Datenbasis

- **1 Jahr Daten** (Jan-Dez 2025) -- keine jahresuebergreifende Validierung moeglich
- **Nur Wochentage + Samstage** in den Daten -- Sonntage fehlen oder haben 0 Belegung
- **Auslastung selten > 30** (nur 49 von 5.997 OOS-Samples) -- Modellverhalten bei hoher Auslastung unsicher

### 8.3 Quantil-Kalibrierung

Die Coverage (87.3%) liegt unter dem theoretischen Ziel von 94%. Dies bedeutet, die Prediction Intervals sind leicht zu eng. Fuer die Produktivnutzung ist dies akzeptabel (>= 85%), aber eine Nachkalibrierung koennte die Zuverlaessigkeit erhoehen.

---

## 9. GO/NO-GO Checkliste

| # | Kriterium | Status |
|---|-----------|--------|
| 1 | MAE besser als alle Baselines | PASS |
| 2 | Improvement >= 8% vs Persistence H60 | PASS (85.8%) |
| 3 | Coverage94 in [85%, 98%] | PASS (87.3%) |
| 4 | DM-Test signifikant vs Persistence (p < 0.05) | PASS (p < 0.0001) |
| 5 | Bootstrap-CI schliesst 0 Verbesserung aus | PASS |
| 6 | Fold-CV < 15% | PASS (7.5%) |
| 7 | Mittlerer Residual (Bias) < 0.5 | PASS (-0.003) |
| 8 | Kein extremes Segment (MAE < 5.0) | PASS |

**Gesamtbewertung: GO**

---

## 10. Empfehlungen

### Sofort (vor Produktivfreigabe)
- Dry Run: bestanden (40/40)
- Feature-Alignment: bekanntes Risiko, dokumentiert, Mapping implementiert

### Kurzfristig (nach Produktivfreigabe)
1. Live-Monitoring der Prognosequalitaet (MAE, Coverage) einrichten
2. Retraining mit `features.py`-Features um Mismatch zu eliminieren
3. Quantil-Nachkalibrierung (Conformal Prediction) pruefen

### Mittelfristig
1. Jahresuebergreifende Validierung mit Daten aus 2026
2. Online-Learning / regelmaessiges Retraining
3. Segment-spezifische Modelle fuer High-Load-Perioden
