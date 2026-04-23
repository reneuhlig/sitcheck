# Explainability V2

## Ziel
Explainability V2 vereinheitlicht den LLM-Eingangskontext (ECP v2), Prompt-Templates und Narrative-Generierung zentral im API-Gateway.

## Pipeline
1. API baut ECP v2 aus Forecast/XAI/Recommendations/History/Lecture-Impact.
2. ECP v2 enthaelt zusaetzlich professorentaugliche Felder: `zone_capacity`, `utilization_now_pct`, `occupancy_explainer`, `improvement_candidates`.
3. Prompt-Registry rendert deterministisch: `system + audience + task + output_rules + context_json`.
4. Ollama wird über `/api/generate` aufgerufen (optional).
5. Output wird validiert (`narrative` + `structured`).
6. Bei Fehlern greift deterministischer Template-Fallback.

## Endpunkte
- `GET /api/v1/explain/context`
- `POST /api/v1/explain/narrative`
- `POST /api/v1/explain/prompt/preview` (nur wenn `EXPLAINABILITY_DEBUG_PROMPT_PREVIEW=true`)

`POST /api/v1/explain/narrative` akzeptiert optional `require_ollama=true`. In diesem Modus wird keine Template-Antwort ausgeliefert: Wenn Ollama/Qwen nicht erreichbar ist oder die LLM-Antwort den Vertrag nicht erfüllt, antwortet der Endpoint mit `502`.

## Verträge
- Kontext: `packages/shared/schemas/llm-explainability-context-v2.schema.json`
- Antwort: `packages/shared/schemas/llm-explanation-response.schema.json`

## Prompt-Templates
- Manifest: `packages/shared/prompts/explainability/manifest.json`
- Assets: `packages/shared/prompts/explainability/*.md`
- Audiences: `ops`, `executive`, `enduser`, `professor`

## Response Envelope (`/api/v1/explain/narrative`)
- `mode`: `ollama` | `template` | `template_fallback`
- `narrative_markdown`
- `structured`
- `response` (vollständiger Dual-Output)
- `context` (ECP v2)
- `meta`: `explain_run_id`, `template_set_id`, `prompt_version`, `context_hash`, `model`, `latency_ms`, `fallback_reason`
- `warnings`

Bei Dashboard-Assistant-Anfragen mit aktivem Ollama muss `mode` `ollama` oder `ollama_hybrid` sein und `meta.model` das verwendete Ollama-Modell enthalten.

## Betriebsparameter
- `EXPLAINABILITY_TEMPLATE_SET`
- `EXPLAINABILITY_RESPONSE_MODE`
- `EXPLAINABILITY_OLLAMA_TIMEOUT_SECONDS`
- `EXPLAINABILITY_FALLBACK_ENABLED`
- `EXPLAINABILITY_DEBUG_PROMPT_PREVIEW`
- `EXPLAINABILITY_PROFESSOR_MODE_ENABLED`
- `OLLAMA_ENABLED`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`

## Observability
Es werden narrative Laufdaten protokolliert:
- Laufzeit (`latency_ms`)
- Parse-Erfolg
- Fallback-Grund
- Evidence-Coverage
