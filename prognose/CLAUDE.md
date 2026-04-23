# Sitcheck Prognose -- Claude Context

Dieses Dokument gibt Claude in neuen Sitzungen sofort den vollen Projektkontext.

---

## Projekt-Identitaet

**Was:** Occupancy-Intelligence-Plattform fuer DHBW Mannheim -- Echtzeit-Belegungsprognosen fuer Bibliotheks-/Lernraeume.
**Wo:** `/sitcheck/prognose` (eigenstaendiges Git-Repo, Submodul in `/sitcheck`).
**Wer:** kiadmin (Student DHBW Mannheim, s232657@student.dhbw-mannheim.de).
**Stand:** Maerz 2026, aktive Entwicklung.

---

## Schnelluebersicht Architektur

13 Docker-Services (Python FastAPI + Node.js MCP), Microservice-Architektur:

| Service | Port | Kern-Datei | Aufgabe |
|---------|------|-----------|---------|
| api-gateway | 8000 | `apps/api-gateway/main.py` | REST API, DB, Orchestrierung, ECP v2 |
| forecast | 8001 | `services/forecast/main.py` | ML-Prognosen, Training, Inference |
| xai | 8002 | `services/xai/main.py` | Erklaerungen (Drivers, Uncertainty) |
| recommendations | 8003 | `services/recommendations/main.py` | Regelwerk, Szenarien |
| forecast-scheduler | 8011 | `services/forecast-scheduler/main.py` | Snapshot-Erzeugung (alle 5 Min) |
| forecast-trainer | 8013 | `services/forecast-trainer/main.py` | Naechtliche Evaluation + Ablation |
| calendar-ingest | 8010 | `services/calendar-ingest/main.py` | ICS-Import |
| lecture-ingest | 8012 | `services/lecture-ingest/main.py` | DHBW Vorlesungs-API |
| dashboard | 8501 | `apps/dashboard/app.py` | Streamlit UI |
| mcp-sitcheck | 8081 | `apps/mcp-sitcheck/server.mjs` | MCP Tool Server (Read-Only) |

**Datenbank:** TimescaleDB (PG 16) produktiv, SQLite lokal.
**Tech:** Python 3.11+, FastAPI, TensorFlow 2.20, scikit-learn, Streamlit, Plotly, LangChain, Ollama (optional), Node.js MCP, AJV, Pydantic, SQLAlchemy.

---

## Prognosemodell -- Aktueller Stand

### Primaermodell: LightGBM Quantile Regression (NEU, Maerz 2026)
- **Modell:** LightGBM mit 3 separaten Quantil-Modellen (q03, q50, q97)
- **Definition:** `services/forecast/model_gbdt.py`
- **Training:** `services/forecast/train_gbdt.py`, Excel-basiert, Walk-Forward 6-Fold
- **Features:** `services/forecast/features_excel.py`, 40 Features, Version `excel_v1`
- **Daten:** `KI_Projekt_Daten_einJahr.xlsx` (13.475 Zeilen, 15-Min, 2025) + DHBW Lecture-Profil
- **Datenaufbereitung:** `scripts/data/prepare_training_data.py` -> `training_data.parquet`
- **Explainability:** SHAP TreeExplainer (`services/xai/shap_explainer.py`), deutsche Labels (`services/xai/feature_labels.py`)

### Performance (6-Fold Walk-Forward CV)
- **MAE GBDT: 0.820** (Baseline: 3.920, Verbesserung: 78.5%)
- **Coverage90: 87.8%** (im Zielkorridor 85-95%)
- **Promotion Gate: PASS**
- Gespeichert unter: `services/forecast/models/default-zone/h60/`

### Challenger: TF-MLP v2
- Single-Output, 3 Layer (256->128->64), BatchNorm, Dropout(0.2), Gradient Clipping
- Definition: `services/forecast/model_tf.py:build_mlp_v2()`
- Altes Multi-Step-MLP bleibt als `build_mlp_model()` fuer Abwaertskompatibilitaet

### Legacy TF-MLP v1 (alt, underperformed)
- MAE 10.60, 7.42x schlechter als Baseline
- 6 Ursachen dokumentiert in `docs/model-evaluation-report.md`
- Wird nicht mehr produktiv genutzt

### Backend-Switch
`FORECAST_MODEL_BACKEND` in .env: `lgbm` (neu, default empfohlen) | `tf_mlp` | `baseline`

---

## Feature-Engineering (40 Features, excel_v1)

| Kategorie | Count | Beispiele |
|-----------|-------|-----------|
| Excel-Faktoren | 7 | `f_month`, `f_weekday`, `f_tod`, `f_weather`, `f_bridge`, `efficiency`, `capacity_effective` |
| Binaer | 5 | `bridge_day`, `winter_break`, `weather_rainy`, `weather_sunny`, `is_partial_closure` |
| Auslastung | 1 | `utilization_pct` |
| Zeitlich/Zyklisch | 8 | `minute_sin/cos`, `dow_sin/cos`, `day_of_year_sin/cos`, `hour_of_day`, `is_weekday` |
| Occupancy-Lags | 8 | `occupancy_lag_{1,2,3,4,8,16}`, `occupancy_lag_day`, `occupancy_lag_week` |
| Rolling Stats | 5 | `occupancy_roll_mean_{4,8,16}`, `occupancy_roll_std_{4,8}` |
| Diffs | 2 | `occupancy_diff_1`, `occupancy_diff_4` |
| Vorlesungs-Proxies | 4 | `lecture_density_proxy`, `lecture_starts/ends/heavy_proxy` |
| Kalender | 2 | `event_active`, `event_impact_sum` |
| Vorlesungen | 11 | `lecture_count_now`, `lecture_net_pull`, `lecture_heavy_now`, ... |
| Datenqualitaet | 3 | `quality_score`, `quality_flag_count`, `utilization` |

---

## Wichtige Dateipfade

### Prognose-Kern
- `services/forecast/features.py` -- Feature Engineering, `build_feature_frame()`, `build_supervised_dataset()`
- `services/forecast/model_tf.py` -- `build_mlp_model()`, MLP Definition
- `services/forecast/train_tf.py` -- `train_zone_model()`, Trainings-Pipeline
- `services/forecast/scientific_eval.py` -- Rolling-Origin Evaluation, Baseline, Promotion Gate
- `services/forecast/main.py` -- FastAPI Forecast Service, Inference, Train-Endpoints
- `services/forecast/model_store.py` -- `save_bundle()`, `load_bundle()`
- `services/forecast/weekly.py` -- Woechentliche Slot-Prognosen
- `services/forecast/trainer_main.py` -- Trainer Entry Point
- `services/forecast/models/` -- Gespeicherte TensorFlow-Modelle

### Explainability
- `apps/api-gateway/explainability/context_builder.py` -- ECP v2 Context
- `apps/api-gateway/explainability/narrative_service.py` -- LLM/Template Narrativ
- `apps/api-gateway/explainability/prompt_registry.py` -- Template-Versionierung
- `services/xai/main.py` -- XAI FastAPI Service
- `packages/shared/prompts/explainability/` -- Prompt-Templates (ops/executive/enduser/professor)

### Schemas und Contracts
- `packages/shared/schemas/` -- 18 kanonische JSON Schemas (Source of Truth)
- `packages/shared/schemas/evidence.schema.json` -- Evidence/Lineage Schema
- `packages/shared/schemas/forecast-response.schema.json` -- Prognose-Antwort
- `packages/shared/schemas/llm-explanation-response.schema.json` -- LLM-Erklaerung

### Datenbank
- `infra/db/migrations/001_init.sql` -- Kern-Schema (zones, counts, forecasts, explanations, ...)
- `infra/db/migrations/003_lecture_activity.sql` -- Vorlesungsdaten
- `infra/db/migrations/004_lineage_and_weekly.sql` -- Model Lineage, Weekly
- `infra/db/migrations/005_auth_and_bookings.sql` -- Auth + Buchungen
- `runtime_local.db` -- Haupt-SQLite (240MB, enthaelt Produktionsdaten)

### Dashboard
- `apps/dashboard/app.py` -- Streamlit Main (Command Center, Forecast Lab, Assistant)
- `apps/dashboard/ui/panels.py` -- UI-Panels (Charts, KPIs, Drivers, Evidence)
- `apps/dashboard/api_client.py` -- HTTP-Client zum API Gateway

### Konfiguration
- `.env.example` -- Alle Umgebungsvariablen mit Defaults
- `docker-compose.yml` -- 13 Services, Profile: dev, calendar, ml-train, ollama
- `pyproject.toml` -- Python-Metadaten
- `package.json` -- Node.js-Metadaten

### Tests
- `scripts/tests/` -- 12 Test-Dateien (Contract, Smoke, Unit, E2E)
- `scripts/train/` -- 4 Convenience-Skripte (eval, promote, switch, retrain)

### Dokumentation (von uns erstellt, Maerz 2026)
- `docs/model-evaluation-report.md` -- Detaillierte Modellbewertung mit 6 Kernproblemen
- `docs/improvement-roadmap.md` -- 14 priorisierte Verbesserungen (Quick Win / Mittel / Strategisch)
- `docs/explainability-assessment.md` -- Explainability-Analyse und Empfehlungen

---

## Priorisierte naechste Schritte (aus improvement-roadmap.md)

1. **QW1+QW3 (KRITISCH, 1-2 Tage):** Horizon-Output auf Terminal-Wert + Evaluations-Paritaet -> beseitigt Evaluations-Artefakt
2. **QW2+QW4 (HOCH, 1 Tag):** Feature-Scaling differenzieren + Target-Clipping
3. **MT2 (HOCH, 1 Woche):** LightGBM/XGBoost als Challenger (GBDTs oft besser auf tabellarischen Daten)
4. **MT1 (HOCH, 1 Woche):** Temporale Architektur (1D-CNN oder LSTM)
5. **MT4+MT5 (MITTEL, 2 Wochen):** Feature Selection + Quantile Regression

---

## Konventionen

- **Sprache:** Code + Commits auf Englisch, Docs auf Deutsch
- **Commit-Style:** `feat(scope): message`, `fix(scope): message`, `chore: message`
- **Schema-First:** Aenderungen an Datenstrukturen zuerst in `packages/shared/schemas/`
- **Evidence-First:** Jeder Datenpunkt traegt Lineage-Metadaten
- **Promotion-Gate:** Kein Modell wird produktiv ohne scientific_pass=true
- **Training standardmaessig gesperrt:** `FORECAST_TRAINING_MODE=locked`
- **Docker-Profile:** `dev` (Demo), `calendar` (ICS), `ml-train` (Trainer), `ollama` (LLM)

---

## Git-Kontext

- **Hauptrepo:** `/sitcheck` (Branch: `livefeed-simulation`)
- **Prognose-Subrepo:** `/sitcheck/prognose` (eigenes .git, Branch: `main`)
- **Achtung:** `prognose` hat KEINE `.gitmodules`-Datei -- es ist ein eingebettetes Repo, kein sauberes Submodul
- **Uncommittete Aenderungen:** Viele Dateien in `prognose` sind unstaged (siehe `git status` im Subrepo)
