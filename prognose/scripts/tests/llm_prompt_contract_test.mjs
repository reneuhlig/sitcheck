import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import Ajv2020 from "ajv/dist/2020.js";
import addFormats from "ajv-formats";
import {
  buildExplainabilityContext,
  renderTemplateFallback,
  validateDualResponseShape,
} from "../../apps/mcp-sitcheck/explainability.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const schemaDir = path.resolve(__dirname, "../../packages/shared/schemas");

const ajv = new Ajv2020({ allErrors: true, strict: false });
addFormats(ajv);

for (const file of readdirSync(schemaDir)) {
  if (!file.endsWith(".schema.json")) continue;
  const schema = JSON.parse(readFileSync(path.join(schemaDir, file), "utf8"));
  ajv.addSchema(schema, schema.$id);
}

const validateContext = ajv.getSchema("https://sitcheck.dev/schemas/llm-explainability-context.schema.json");
const validateResponse = ajv.getSchema("https://sitcheck.dev/schemas/llm-explanation-response.schema.json");

function ensure(validator, payload, label) {
  if (!validator) throw new Error(`missing validator: ${label}`);
  if (!validator(payload)) throw new Error(`${label} failed: ${ajv.errorsText(validator.errors)}`);
}

const evidence = {
  evidence_id: "ev-1",
  generated_at: "2026-02-24T10:00:00Z",
  time_window: { from: "2026-02-24T09:00:00Z", to: "2026-02-24T10:00:00Z" },
  sources: [{ type: "counts", id: "window-1" }],
  model: { name: "baseline", version: "v1", backend: "baseline" },
  quality: { score: 0.9, flags: ["OK"] },
};

const context = buildExplainabilityContext({
  forecastLatest: {
    zone_id: "default-zone",
    horizon: 60,
    generated_at: "2026-02-24T10:00:00Z",
    age_seconds: 15,
    stale: false,
    summary: "Stable trend",
    model_version: "baseline-v1",
    points: [{ timestamp: "2026-02-24T10:01:00Z", yhat: 35, pi_low: 30, pi_high: 40 }],
    evidence,
    source: "snapshot",
  },
  explanation: {
    zone_id: "default-zone",
    horizon: 60,
    summary: "Momentum-driven",
    drivers: [{ name: "momentum", impact: 0.3, direction: "up", description: "recent trend" }],
    uncertainty: { score: 0.2, level: "low", reason: "enough data" },
    evidence,
  },
  recommendation: {
    zone_id: "default-zone",
    horizon: 60,
    summary: "Monitor",
    actions: [{ action_type: "monitor", priority: 1, rationale: "No risk", expected_impact: { delta_occupancy: -2 } }],
    gates: { quality_ok: true, uncertainty_ok: true, notes: [] },
    evidence,
  },
  history: {
    points: [
      {
        timestamp: "2026-02-24T10:00:00Z",
        zone_id: "default-zone",
        occupancy: 35,
        utilization: 0.35,
        quality_score: 0.9,
        quality_flags: ["OK"],
      },
    ],
  },
  audience: "executive",
  language: "de",
  timezone: "UTC",
  query: "Was ist die Begründung der Prognose?",
});

ensure(validateContext, context, "llm context schema");

const out1 = renderTemplateFallback(context);
const out2 = renderTemplateFallback(context);
if (JSON.stringify(out1) !== JSON.stringify(out2)) {
  throw new Error("template fallback is not deterministic");
}

ensure(validateResponse, out1, "llm response schema");
const shape = validateDualResponseShape(out1);
if (!shape.ok) throw new Error(`dual response shape invalid: ${shape.reason}`);

console.log("llm prompt contract tests passed");
