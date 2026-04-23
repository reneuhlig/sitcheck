import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import Ajv2020 from "ajv/dist/2020.js";
import addFormats from "ajv-formats";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const schemaDir = path.resolve(__dirname, "../../packages/shared/schemas");

const ajv = new Ajv2020({ allErrors: true, strict: false });
addFormats(ajv);

for (const file of readdirSync(schemaDir)) {
  if (!file.endsWith(".schema.json")) continue;
  const schema = JSON.parse(readFileSync(path.join(schemaDir, file), "utf8"));
  ajv.addSchema(schema, schema.$id);
}

const evidence = {
  evidence_id: "ev-1",
  generated_at: "2026-02-18T19:00:00Z",
  time_window: { from: "2026-02-18T18:00:00Z", to: "2026-02-18T19:00:00Z" },
  sources: [{ type: "counts", id: "window-1" }],
  model: { name: "baseline", version: "v1" },
  quality: { score: 0.92, flags: ["OK"] }
};

const forecastLatest = {
  zone_id: "default-zone",
  horizon: 60,
  generated_at: "2026-02-18T19:00:00Z",
  age_seconds: 12,
  stale: false,
  summary: "stable",
  model_version: "v1",
  points: [{ timestamp: "2026-02-18T19:01:00Z", yhat: 30, pi_low: 25, pi_high: 35 }],
  evidence,
  source: "snapshot",
};

const explanation = {
  zone_id: "default-zone",
  horizon: 60,
  summary: "momentum-driven",
  drivers: [{ name: "momentum", impact: 0.3, direction: "up", description: "recent trend" }],
  uncertainty: { score: 0.2, level: "low", reason: "enough data" },
  evidence,
};

const recommendation = {
  zone_id: "default-zone",
  horizon: 60,
  summary: "open room",
  actions: [{ action_type: "open_room", priority: 1, rationale: "peak", expected_impact: { delta_occupancy: -12 } }],
  gates: { quality_ok: true, uncertainty_ok: true, notes: [] },
  evidence,
};

const commandCenter = {
  meta: {
    generated_at: "2026-02-18T19:00:00Z",
    zone_id: "default-zone",
    horizon: 60,
    history_minutes: 180,
    long_term_days: 14,
    stale_seconds: 900,
    environment: "dev",
  },
  service_health: [
    { service: "api-gateway", status: "ok", latency_ms: 0, detail: "local" },
    { service: "forecast", status: "ok", latency_ms: 10, detail: "ok" },
  ],
  live: {
    timestamp: "2026-02-18T19:00:00Z",
    occupancy: 42,
    utilization: 0.42,
    quality_score: 0.9,
    quality_flags: ["OK"],
    point_count: 180,
  },
  history: {
    zone_id: "default-zone",
    from: "2026-02-18T16:00:00Z",
    to: "2026-02-18T19:00:00Z",
    granularity: "1m",
    points: [{ timestamp: "2026-02-18T19:00:00Z", zone_id: "default-zone", occupancy: 30, utilization: 0.3, quality_flags: ["OK"], evidence }],
  },
  forecast_latest: forecastLatest,
  forecast_long_term: [forecastLatest],
  explanation,
  recommendations: recommendation,
  calendar_events: [
    {
      event_id: "evt-1",
      zone_id: "default-zone",
      title: "Vorlesungswechsel",
      starts_at: "2026-02-18T20:00:00Z",
      ends_at: "2026-02-18T21:00:00Z",
      source: "mock",
      metadata: {},
    },
  ],
  alerts: [{ code: "ALL_CLEAR", level: "ok", message: "Keine kritischen Warnungen." }],
};
const lectureActivity = {
  zone_id: "default-zone",
  from: "2026-02-18T16:00:00Z",
  to: "2026-02-18T19:00:00Z",
  granularity: "1m",
  points: [
    {
      timestamp: "2026-02-18T19:00:00Z",
      zone_id: "default-zone",
      active_lectures: 14,
      active_courses: 10,
      starts_next_60m: 22,
      ends_next_60m: 17,
      source: "rapla_refresh",
      quality_score: 0.9,
      quality_flags: ["LECTURE_RAPLA"],
      metadata: { site_code: "MA" },
    },
  ],
};

const cases = [
  ["https://sitcheck.dev/schemas/zone.schema.json", { zone_id: "default-zone", name: "Main", capacity: 100, is_active: true }],
  ["https://sitcheck.dev/schemas/count-point.schema.json", { timestamp: "2026-02-18T19:00:00Z", zone_id: "default-zone", occupancy: 30, utilization: 0.3, quality_flags: ["OK"], evidence }],
  ["https://sitcheck.dev/schemas/forecast-response.schema.json", { zone_id: "default-zone", horizon: 60, generated_at: "2026-02-18T19:00:00Z", summary: "stable", model_version: "v1", points: [{ timestamp: "2026-02-18T19:01:00Z", yhat: 30, pi_low: 25, pi_high: 35 }], evidence }],
  ["https://sitcheck.dev/schemas/forecast-latest-response.schema.json", forecastLatest],
  ["https://sitcheck.dev/schemas/explanation.schema.json", explanation],
  ["https://sitcheck.dev/schemas/recommendation.schema.json", recommendation],
  ["https://sitcheck.dev/schemas/scenario-input.schema.json", { zone_id: "default-zone", horizon: 60, persist: false, changes: { open_room: true } }],
  ["https://sitcheck.dev/schemas/scenario-result.schema.json", { zone_id: "default-zone", horizon: 60, summary: "reduced peak", baseline: { peak_occupancy: 90, peak_utilization: 0.9 }, counterfactual: { peak_occupancy: 80, peak_utilization: 0.8 }, delta: { peak_occupancy: -10, peak_utilization: -0.1 }, evidence }],
  ["https://sitcheck.dev/schemas/dashboard-command-center.schema.json", commandCenter],
  ["https://sitcheck.dev/schemas/lecture-activity-response.schema.json", lectureActivity],
  [
    "https://sitcheck.dev/schemas/llm-explainability-context.schema.json",
    {
      forecast_latest: forecastLatest,
      explanation,
      recommendation,
      history_digest: {
        last_occupancy: 42,
        last_utilization: 0.42,
        trend_15m: 1.5,
        trend_60m: 3.1,
        quality_score_avg: 0.9,
        quality_flags_top: ["OK"],
        point_count: 120,
      },
      scenario_digest: null,
      context_meta: {
        request_id: "ctx-1",
        generated_at: "2026-02-18T19:00:00Z",
        audience: "ops",
        language: "de",
        timezone: "UTC",
        guardrails: ["api_tools_only", "no_invented_facts"],
        query: "Warum steigt die Auslastung?",
      },
      citation_map: [
        {
          ref_id: "ref-1",
          evidence_id: "ev-1",
          source_type: "counts",
          source_id: "window-1",
          time_window: { from: "2026-02-18T18:00:00Z", to: "2026-02-18T19:00:00Z" },
          model_version: "v1",
          quality_score: 0.9,
          quality_flags: ["OK"],
        },
      ],
    },
  ],
  [
    "https://sitcheck.dev/schemas/llm-explanation-response.schema.json",
    {
      narrative: {
        one_liner: "Prognose bleibt stabil.",
        warum: "Momentum ist positiv [ref:ref-1].",
        unsicherheit: "Unsicherheit ist niedrig [ref:ref-1].",
        empfehlung: "Weiter beobachten [ref:ref-1].",
        evidence_hinweis: "[ref:ref-1]",
      },
      structured: {
        audience: "professor",
        zone_id: "default-zone",
        horizon: 60,
        verdict: "monitor",
        top_drivers: [{ name: "momentum", impact: 0.3, direction: "up", evidence_ref: "ref-1" }],
        uncertainty: { score: 0.2, level: "low", reason: "Sufficient history", evidence_ref: "ref-1" },
        recommended_actions: [
          { action_type: "monitor", priority: 1, rationale: "No severe risk", evidence_ref: "ref-1" },
        ],
        evidence_refs: ["ref-1"],
        confidence_statement: "High confidence",
        limitations: ["None"],
      },
    },
  ],
  [
    "https://sitcheck.dev/schemas/llm-explainability-context-v2.schema.json",
    {
      request_meta: {
        request_id: "ctx-v2-1",
        generated_at: "2026-02-18T19:00:00Z",
        zone_id: "default-zone",
        horizon: 60,
        audience: "professor",
        language: "de",
        query: "Warum steigt die Auslastung?",
        guardrails: ["api_tools_only", "no_invented_facts"],
        template_set_id: "explainability-de-v2",
        prompt_version: "2.0.0",
      },
      zone_capacity: 100,
      utilization_now_pct: 42.0,
      occupancy_explainer: {
        current_occupancy: 42,
        avg_60m: 39.8,
        trend_15m: 1.5,
        forecast_next_60m_peak: 48.0,
        risk_level: "medium",
      },
      improvement_candidates: [
        {
          measure_id: "capacity-buffer-next-slot",
          measure_text: "Pufferbereich fuer Peak vorbereiten.",
          expected_effect: "Spitzenlast abfedern.",
          effort: "low",
          owner_hint: "Bibliotheksteam",
          evidence_ref: "ref-1",
        },
        {
          measure_id: "lecture-transition-communication",
          measure_text: "Vorlesungswechsel kommunizieren.",
          expected_effect: "Zufluss verteilen.",
          effort: "low",
          owner_hint: "Service Desk",
          evidence_ref: "ref-1",
        },
        {
          measure_id: "quality-hardening",
          measure_text: "Datenqualitaet im Betrieb pruefen.",
          expected_effect: "Belastbarere Prognosen.",
          effort: "medium",
          owner_hint: "Data/IT Betrieb",
          evidence_ref: "ref-1",
        },
      ],
      forecast_snapshot: {
        zone_id: "default-zone",
        horizon: 60,
        generated_at: "2026-02-18T19:00:00Z",
        age_seconds: 12,
        stale: false,
        summary: "stable",
        model_version: "v1",
        source: "snapshot",
        next_point: { timestamp: "2026-02-18T19:01:00Z", yhat: 30, pi_low: 25, pi_high: 35 },
        peak: { value: 30, timestamp: "2026-02-18T19:01:00Z" },
        point_count: 1,
        evidence_ref: "ref-1",
      },
      history_digest: {
        last_occupancy: 42,
        last_utilization: 0.42,
        trend_15m: 1.5,
        trend_60m: 3.1,
        quality_score_avg: 0.9,
        quality_flags_top: ["OK"],
        point_count: 120,
        similar_pattern: { found: false, reason: "insufficient_history" },
      },
      driver_summary: {
        top_drivers: [
          { name: "momentum", impact: 0.3, direction: "up", description: "recent trend", evidence_ref: "ref-1" },
        ],
        evidence_ref: "ref-1",
      },
      uncertainty: { score: 0.2, level: "low", reason: "enough data", evidence_ref: "ref-1" },
      recommendation_digest: {
        summary: "monitor",
        quality_ok: true,
        uncertainty_ok: true,
        actions: [{ action_type: "monitor", priority: 1, rationale: "No risk", evidence_ref: "ref-1" }],
        evidence_ref: "ref-1",
      },
      lecture_impact_digest: {
        active_lectures: 14,
        active_courses: 10,
        starts_next_60m: 22,
        ends_next_60m: 17,
        heavy_active_lectures: 2,
        heavy_ended_last_60m: 1,
        lecture_pull_regular: 280.0,
        heavy_bib_bonus: 12.0,
        lecture_net_pull: 268.0,
        impact_model_version: "lecture-impact-v1",
      },
      quality_digest: {
        live_quality_score: 0.9,
        live_quality_flags: ["OK"],
        history_quality_score_avg: 0.9,
        live_point_count: 180,
        alerts: [{ code: "ALL_CLEAR", level: "ok", message: "Keine kritischen Warnungen." }],
      },
      citation_map: [
        {
          ref_id: "ref-1",
          evidence_id: "ev-1",
          source_type: "counts",
          source_id: "window-1",
          time_window: { from: "2026-02-18T18:00:00Z", to: "2026-02-18T19:00:00Z" },
          model_version: "v1",
          quality_score: 0.9,
          quality_flags: ["OK"],
        },
      ],
      policy_block: {
        abstain_rules: ["Nur monitor bei hoher Unsicherheit."],
        decision_gates: { quality_ok: true, uncertainty_ok: true, forecast_stale: false },
        output_contract: {
          narrative_keys: ["one_liner", "warum", "unsicherheit", "empfehlung", "evidence_hinweis"],
          structured_required: [
            "audience",
            "zone_id",
            "horizon",
            "verdict",
            "top_drivers",
            "uncertainty",
            "recommended_actions",
            "evidence_refs",
            "confidence_statement",
            "limitations",
          ],
        },
      },
    },
  ],
];

for (const [schemaId, data] of cases) {
  const validate = ajv.getSchema(schemaId);
  if (!validate) throw new Error(`Schema not loaded: ${schemaId}`);
  const ok = validate(data);
  if (!ok) throw new Error(`Validation failed for ${schemaId}: ${ajv.errorsText(validate.errors)}`);
}

const invalidZone = { zone_id: "bad", name: "bad", is_active: true };
const validateZone = ajv.getSchema("https://sitcheck.dev/schemas/zone.schema.json");
if (validateZone(invalidZone)) {
  throw new Error("Invalid zone should have failed AJV validation");
}

console.log("node ajv schema contract tests passed");
