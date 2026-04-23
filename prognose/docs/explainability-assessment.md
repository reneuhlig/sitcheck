# Explainability Assessment: Sitcheck Prognosemodell

**Stand:** 2026-03-28
**Bezug:** [Model Evaluation Report](./model-evaluation-report.md), [Improvement Roadmap](./improvement-roadmap.md)

---

## 1. Uebersicht

Das Sitcheck-System verfuegt ueber ein mehrschichtiges Explainability-System:
- **XAI-Service** (`services/xai/main.py`) -- Numerische Erklaerungen (Drivers, Uncertainty, Evidence)
- **ECP v2** (`apps/api-gateway/explainability/`) -- LLM-basierte Narrativ-Generierung
- **Prompt-Templates** (`packages/shared/prompts/explainability/`) -- 4 Zielgruppen
- **Evidence-Architektur** -- Jeder Datenpunkt traegt Lineage-Metadaten

---

## 2. Bewertung des aktuellen Stands

### 2.1 Staerken

#### Evidence-First-Architektur (Sehr gut)
Jede Erklaerung, Empfehlung und Prognose traegt ein `evidence`-Objekt mit:
- `evidence_id`, `generated_at`, `time_window`
- `sources` (Typ + ID fuer jede Datenquelle)
- `model` (Name, Version, Backend)
- `quality` (Score 0-1, Flags)

**Schema:** `packages/shared/schemas/evidence.schema.json`

Dies ermoeglicht vollstaendige Nachvollziehbarkeit und ist selten in Produktionssystemen zu finden.

#### Progressive Disclosure (Sehr gut)
Das System implementiert 4 Erklaerungsebenen:
1. **One-Liner:** Einzeilige Zusammenfassung
2. **Top Drivers:** Die 3 wichtigsten Einflussfaktoren mit Richtung und Impact
3. **Evidence/Citations:** Vollstaendige Quellenangaben fuer jede Behauptung
4. **Counterfactual Simulation:** "Was waere wenn?"-Szenarien (`POST /api/v1/scenarios/simulate`)

Dies folgt dem Usable XAI Pattern: Nutzer koennen bei Bedarf tiefer einsteigen.

#### Versionierte Prompt-Templates (Gut)
- 4 Zielgruppen: `ops`, `executive`, `enduser`, `professor`
- Sprachversion: Deutsch (`explainability-de-v2`)
- Manifest-basierte Versionierung: `packages/shared/prompts/explainability/manifest.json`
- Dual-Output-Vertrag: Narrativ-Felder + strukturierter JSON-Block

#### Dual Output Contract (Gut)
Jede LLM-Erklaerung liefert sowohl:
1. **Narrativ:** `one_liner`, `warum`, `unsicherheit`, `empfehlung`, `evidence_hinweis`
2. **Strukturiert:** JSON mit Evidence-Referenzen fuer jede Behauptung

**Schema:** `packages/shared/schemas/llm-explanation-response.schema.json`

#### Booking-Overlay-Transparenz (Gut)
Wenn Buchungen in Prognosen einfliessen, wird dies explizit in der Evidence dokumentiert:
- Source-Type: `bookings:overlay`
- Metadata: `count`, `peak_increment`

### 2.2 Luecken

#### SHAP deaktiviert
**Status:** `XAI_SHAP_ENABLED=false` (Standard-Konfiguration)

SHAP (SHapley Additive exPlanations) ist die Standard-Methode fuer modellunabhaengige Feature-Attribution. Ohne SHAP basieren die "Drivers" auf Heuristiken statt auf mathematisch fundierten Attributionen.

**Empfehlung:** SHAP fuer promoted Modelle aktivieren (globale Summary + lokale Einzelerklaerungen).

#### Keine Feature Importance in Evaluation-Reports
Die wissenschaftlichen Evaluationsberichte (`scientific_eval.py:84-100`) speichern Metriken, aber keine Feature-Importance-Rankings. Ohne diese ist unklar, welche der 37 Features tatsaechlich zur Prognose beitragen.

**Empfehlung:** Permutation Importance nach jedem Trainingsrun berechnen und im Report persistieren.

#### Fehlende Modell-Ehrlichkeit
Wenn das System auf die Baseline zurueckfaellt (was aktuell immer der Fall ist, da das TF-Modell die Promotion nicht besteht), kommunizieren die Erklaerungen nicht transparent, dass ein einfaches statistisches Modell statt des ML-Modells verwendet wird.

**Empfehlung:** Einen "Model Honesty Layer" einfuehren: Bei Baseline-Fallback explizit "Einfache statistische Prognose (saisonale Baseline)" statt implizierter ML-Sophistikation kommunizieren.

#### Keine Explanation Confidence
Erklaerungen tragen aktuell kein Konfidenz-Mass fuer die Erklaerung selbst. Eine Erklaerung fuer eine hochunsichere Prognose hat dieselbe "Autoritaet" wie eine fuer eine zuverlaessige.

**Empfehlung:** Explanation Confidence Score einfuehren, basierend auf:
- Modell-Performance (MAE relativ zur Baseline)
- Coverage-Rate der Prediction Intervals
- Unsicherheits-Score aus dem XAI-Service
- Datenqualitaet der zugrundeliegenden Counts

---

## 3. Modelltyp und Erklaerbarkeit

### 3.1 Aktuelles Modell: TensorFlow MLP

**Inherent interpretierbar?** Nein. Ein MLP mit 128+64 Hidden Units ist eine Black Box.

**Geeignete Erklaerungsmethoden:**

| Methode | Typ | Nutzen | Implementierungsaufwand |
|---------|-----|--------|------------------------|
| SHAP TreeExplainer | Global + Lokal | Feature-Attribution pro Prognose | Mittel (nur fuer GBDT) |
| SHAP KernelExplainer | Global + Lokal | Universell, aber langsam | Mittel |
| Permutation Importance | Global | Feature-Ranking | Gering |
| PDP (Partial Dependence) | Global | Feature-Response-Kurven | Gering |
| ALE (Accumulated Local Effects) | Global | Korrelationsrobuster als PDP | Mittel |
| LIME | Lokal | Einzelerklaerungen | Mittel |

### 3.2 Empfehlung bei Modellwechsel zu GBDT

Ein LightGBM-Modell (siehe Improvement Roadmap MT2) wuerde die Erklaerbarkeit erheblich verbessern:
- **Native Feature Importance** (Split + Gain)
- **SHAP TreeExplainer** ist exakt und schnell (keine Approximation noetig)
- **Monotonie-Constraints** koennen domaenenspezifisches Wissen erzwingen

---

## 4. Empfehlungen fuer globale Erklaerbarkeit

### 4.1 Feature Importance Dashboard
**Was:** Balkendiagramm der Top-10-Features nach Permutation Importance oder SHAP Summary.
**Wo:** Dashboard -> Forecast Lab Tab
**Nutzen:** Stakeholder sehen auf einen Blick, welche Faktoren die Prognose treiben.
**Risiko:** Ohne Kontext koennen Features missverstanden werden (z.B. "occupancy_lag_1 ist wichtig" sagt nur "letzte Belegung ist relevant").

### 4.2 PDP/ALE-Plots fuer Top-Features
**Was:** Partial Dependence Plots fuer die 5 wichtigsten Features zeigen den marginalen Effekt auf die Prognose.
**Wo:** Dashboard -> Forecast Lab Tab (erweiterbar)
**Nutzen:** Zeigt nichtlineare Zusammenhaenge (z.B. "Ab 50 Personen steigt die Prognose ueberproportional").
**Risiko:** PDP kann bei korrelierten Features irrefuehren; ALE ist robuster.

### 4.3 Vorlesungs-Impact-Transparenz
**Was:** Der bestehende `lecture_impact_summary` in `scientific_eval.py:498-518` liefert bereits:
- `minutes_with_heavy_effect_share`
- `feature_availability_rate`
- `lecture_net_pull` (mean, q10, q50, q90)

**Empfehlung:** Diese Zusammenfassung ins Dashboard integrieren und als "Vorlesungseinfluss" visualisieren.

---

## 5. Empfehlungen fuer lokale Erklaerbarkeit

### 5.1 SHAP fuer Einzelprognosen
**Was:** Fuer jede Prognose die Top-3 Features mit positivem und negativem SHAP-Wert anzeigen.
**Wo:** `/api/v1/explain` Endpoint erweitern
**Nutzen:** "Diese Prognose ist hoeher als ueblich, weil: (1) Vorlesung Machine Learning gerade laeuft (+12), (2) Belegung steigt seit 30 Min (+8), (3) Dienstagmuster (+5)"
**Risiko:** SHAP KernelExplainer ist langsam (~1-5s pro Erklaerung). Fuer Echtzeit besser TreeExplainer mit GBDT nutzen.

### 5.2 Natuerlichsprachliche Einzelerklaerungen
**Was:** Die bestehende ECP v2 Pipeline (`apps/api-gateway/explainability/context_builder.py`) kann SHAP-Werte als zusaetzlichen Kontext aufnehmen, um praezisere Narrativ-Erklaerungen zu generieren.
**Wo:** `occupancy_explainer` Block im ECP v2 Context
**Nutzen:** Statt generischer Drivers ("Trend steigt") spezifische Attributionen ("Vorlesung X erhoeht die Prognose um 12 Personen").

### 5.3 Counterfactual-Erklaerungen
**Was:** Bereits implementiert als Szenario-Simulation (`POST /api/v1/scenarios/simulate`).
**Verbesserung:** Automatische Counterfactuals generieren: "Ohne die aktuelle Vorlesung waere die Prognose 35 statt 47."
**Nutzen:** Intuitiv verstaendlich, auch ohne ML-Wissen.

---

## 6. Fachliche Erklaerungen fuer Stakeholder

### 6.1 Zielgruppen und Erklaerungstiefe

| Zielgruppe | Template | Erklaerungstiefe | Beispiel |
|------------|----------|-----------------|----------|
| Operations | `ops` | Handlungsorientiert | "Raum zu 85% ausgelastet in 30 Min. Empfehlung: Ausweichraum oeffnen." |
| Executive | `executive` | Trend + KPI | "Belegung steigt Di-Do um 15% gegenueber Vorwoche. Treiber: Pruefungsphase." |
| Enduser | `enduser` | Einfach + direkt | "In 30 Minuten wird es voll. Besser jetzt kommen." |
| Professor | `professor` | Methodisch + vollstaendig | "MLP-Prognose basiert auf 37 Features. Haupttreiber: lag_1 (SHAP: 0.42), lecture_net_pull (0.28). Coverage: 91%." |

Die bestehenden 4 Templates sind gut abgestimmt. Verbesserungspotenzial besteht in der **dynamischen Anreicherung mit SHAP-Werten** fuer den Professor-Modus.

### 6.2 Risiken bei Fehlinterpretationen

1. **Korrelation vs. Kausalitaet:** "lecture_net_pull ist der wichtigste Treiber" bedeutet nicht, dass Vorlesungen die Belegung *verursachen* -- sie korrelieren nur.
2. **Feature-Namen sind technisch:** "occupancy_lag_1" ist fuer Laien unverstaendlich. Natuerlichsprachliche Uebersetzungen sind noetig.
3. **Unsicherheit wird unterschaetzt:** Breite Prediction Intervals koennen als Modellversagen interpretiert werden, obwohl sie korrekte Unsicherheit widerspiegeln.
4. **Baseline-Fallback ist intransparent:** Aktuell ist dem Nutzer nicht klar, ob ein ML-Modell oder eine einfache Formel die Prognose erstellt.

---

## 7. Visualisierungsempfehlungen

### 7.1 Existierende Visualisierungen (Dashboard)

| Visualisierung | Datei | Status |
|---------------|-------|--------|
| History + Forecast Corridor Chart | `apps/dashboard/ui/panels.py:render_forecast_chart()` | Implementiert |
| Top Drivers Balkendiagramm | `apps/dashboard/ui/panels.py:render_drivers_and_recommendations()` | Implementiert |
| Weekly Heatmap Grid | `apps/dashboard/ui/panels.py:render_weekly_outlook()` | Implementiert |
| Evidence/Citations Modal | `apps/dashboard/ui/panels.py:render_evidence()` | Implementiert |
| KPI Strip | `apps/dashboard/ui/panels.py:render_kpis()` | Implementiert |

### 7.2 Empfohlene neue Visualisierungen

1. **SHAP Waterfall Plot:** Zeigt pro Prognose den Beitrag jedes Features von der Baseline zum finalen Wert. Ideal fuer den Professor-Modus.
2. **Feature Importance Timeline:** Wie veraendert sich die Wichtigkeit der Top-5-Features ueber die Zeit? Zeigt saisonale Muster.
3. **Modell-Konfidenz-Ampel:** Einfache Ampel (gruen/gelb/rot) basierend auf Uncertainty + Coverage + Modelltyp. Fuer alle Zielgruppen.
4. **Residual-Heatmap:** Fehlerverteilung nach Wochentag und Tageszeit. Zeigt systematische Schwaechen.

---

## 8. Integration in Produktion

### 8.1 Architektur fuer produktive Explainability

```
Prognose-Request
    |
    v
Forecast Service (:8001) --> Prediction + Feature Vector
    |
    v
XAI Service (:8002) --> SHAP Values + Drivers + Uncertainty
    |
    v
API Gateway (:8000) --> ECP v2 Context Builder
    |                       |
    v                       v
Dashboard (:8501)     LLM Narrative (Ollama)
    |                       |
    v                       v
Nutzer sieht:         Nutzer sieht:
- Chart + PI          - Natuerlichsprachlich
- Drivers             - Zielgruppengerecht
- Ampel               - Mit Evidence-Refs
```

### 8.2 Performance-Ueberlegungen

| Methode | Latenz (pro Erklaerung) | Empfehlung |
|---------|------------------------|------------|
| Permutation Importance | ~5s (einmalig pro Training) | Im Trainingsreport speichern |
| SHAP KernelExplainer | 1-5s pro Anfrage | Nur fuer on-demand Erklaerungen |
| SHAP TreeExplainer (GBDT) | <100ms pro Anfrage | Fuer Echtzeit-Erklaerungen |
| PDP/ALE | ~10s (einmalig pro Training) | Vorberechnen und cachen |
| LLM Narrative | 1-10s (je nach Modell) | Template-Fallback bei Timeout |

### 8.3 Empfohlene Priorisierung

1. **Sofort:** Modell-Ehrlichkeit (Baseline vs. ML transparent kommunizieren)
2. **Kurzfristig:** Permutation Importance im Training persistieren
3. **Mittelfristig:** SHAP aktivieren (global Summary + lokale Top-3)
4. **Mittelfristig:** PDP/ALE vorberechnen fuer Top-5 Features
5. **Langfristig:** SHAP TreeExplainer mit GBDT fuer Echtzeit-Erklaerungen
6. **Langfristig:** Explanation Confidence Score in alle Endpoints integrieren
