import { existsSync, readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import Ajv2020 from "ajv/dist/2020.js";
import addFormats from "ajv-formats";
import { fetch } from "undici";
import {
  formatDualOutput,
} from "./explainability.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const schemaDirCandidates = [
  path.resolve(__dirname, "../../packages/shared/schemas"),
  path.resolve(__dirname, "./schemas"),
];
const schemaDir = schemaDirCandidates.find((candidate) => existsSync(candidate));

if (!schemaDir) {
  throw new Error("mcp-sitcheck: shared schema directory not found");
}

function loadAjv() {
  const ajv = new Ajv2020({ allErrors: true, strict: false });
  addFormats(ajv);

  for (const file of readdirSync(schemaDir)) {
    if (!file.endsWith(".schema.json")) continue;
    const schema = JSON.parse(readFileSync(path.join(schemaDir, file), "utf8"));
    ajv.addSchema(schema, schema.$id);
  }

  return ajv;
}

function makeError(message) {
  return new Error(`mcp-sitcheck: ${message}`);
}

class SitcheckApiClient {
  constructor(baseUrl) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  static formatErrorDetail(method, pathname, status, bodyText) {
    let parsed = null;
    try {
      parsed = JSON.parse(bodyText);
    } catch (_err) {
      parsed = null;
    }

    const detail = parsed?.detail;
    if (status === 502 && detail && typeof detail === "object" && detail.code === "llm_quality_gate_failed") {
      return (
        `${method} ${pathname} failed: 502 llm_quality_gate_failed ` +
        `(actual_coverage=${detail.actual_coverage}, min_coverage=${detail.min_coverage}, query_intent=${detail.query_intent})`
      );
    }

    if (detail && typeof detail === "object") {
      return `${method} ${pathname} failed: ${status} ${JSON.stringify(detail)}`;
    }

    if (typeof detail === "string" && detail.trim()) {
      return `${method} ${pathname} failed: ${status} ${detail}`;
    }

    return `${method} ${pathname} failed: ${status} ${bodyText}`;
  }

  async get(pathname, params = {}) {
    const url = new URL(`${this.baseUrl}${pathname}`);
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "") {
        url.searchParams.set(key, String(value));
      }
    });

    const response = await fetch(url);
    if (!response.ok) {
      const bodyText = await response.text();
      throw makeError(SitcheckApiClient.formatErrorDetail("GET", pathname, response.status, bodyText));
    }
    return response.json();
  }

  async post(pathname, body = {}) {
    const response = await fetch(`${this.baseUrl}${pathname}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const bodyText = await response.text();
      throw makeError(SitcheckApiClient.formatErrorDetail("POST", pathname, response.status, bodyText));
    }
    return response.json();
  }
}

function ensure(validator, data, label) {
  const ok = validator(data);
  if (!ok) {
    throw makeError(`${label} validation failed: ${JSON.stringify(validator.errors)}`);
  }
}

export function buildToolRuntime(config = {}) {
  const ajv = loadAjv();
  const api = new SitcheckApiClient(config.apiBaseUrl || process.env.API_BASE_URL || "http://localhost:8000");

  const schemaIds = {
    forecast: "https://sitcheck.dev/schemas/forecast-response.schema.json",
    forecastLatest: "https://sitcheck.dev/schemas/forecast-latest-response.schema.json",
    explanation: "https://sitcheck.dev/schemas/explanation.schema.json",
    recommendation: "https://sitcheck.dev/schemas/recommendation.schema.json",
    scenarioResult: "https://sitcheck.dev/schemas/scenario-result.schema.json",
    calendarEvent: "https://sitcheck.dev/schemas/calendar-event.schema.json",
    countPoint: "https://sitcheck.dev/schemas/count-point.schema.json",
    llmContext: "https://sitcheck.dev/schemas/llm-explainability-context.schema.json",
    llmContextV2: "https://sitcheck.dev/schemas/llm-explainability-context-v2.schema.json",
    llmResponse: "https://sitcheck.dev/schemas/llm-explanation-response.schema.json",
  };

  const validateForecast = ajv.getSchema(schemaIds.forecast);
  const validateForecastLatest = ajv.getSchema(schemaIds.forecastLatest);
  const validateExplanation = ajv.getSchema(schemaIds.explanation);
  const validateRecommendation = ajv.getSchema(schemaIds.recommendation);
  const validateScenarioResult = ajv.getSchema(schemaIds.scenarioResult);
  const validateLlmContext = ajv.getSchema(schemaIds.llmContext);
  const validateLlmContextV2 = ajv.getSchema(schemaIds.llmContextV2);
  const validateLlmResponse = ajv.getSchema(schemaIds.llmResponse);
  const validateCalendarEvent = ajv.compile({
    type: "array",
    items: { $ref: schemaIds.calendarEvent },
  });
  const validateCountPoint = ajv.getSchema(schemaIds.countPoint);

  const inputSchemas = {
    get_live_occupancy: {
      type: "object",
      additionalProperties: false,
      required: ["zone_id"],
      properties: { zone_id: { type: "string" } },
    },
    get_history: {
      type: "object",
      additionalProperties: false,
      required: ["zone_id", "from", "to", "granularity"],
      properties: {
        zone_id: { type: "string" },
        from: { type: "string", format: "date-time" },
        to: { type: "string", format: "date-time" },
        granularity: { type: "string", enum: ["raw", "1m", "5m", "15m"] },
      },
    },
    get_forecast: {
      type: "object",
      additionalProperties: false,
      required: ["zone_id", "horizon"],
      properties: {
        zone_id: { type: "string" },
        horizon: { type: "integer", minimum: 1, maximum: 43200 },
      },
    },
    explain_forecast: {
      type: "object",
      additionalProperties: false,
      required: ["zone_id", "horizon"],
      properties: {
        zone_id: { type: "string" },
        horizon: { type: "integer", minimum: 1, maximum: 720 },
      },
    },
    list_calendar_events: {
      type: "object",
      additionalProperties: false,
      required: ["zone_id", "from", "to"],
      properties: {
        zone_id: { type: "string" },
        from: { type: "string", format: "date-time" },
        to: { type: "string", format: "date-time" },
      },
    },
    recommend_actions: {
      type: "object",
      additionalProperties: false,
      required: ["zone_id", "horizon"],
      properties: {
        zone_id: { type: "string" },
        horizon: { type: "integer", minimum: 1, maximum: 720 },
      },
    },
    simulate_scenario: {
      type: "object",
      additionalProperties: false,
      required: ["zone_id", "horizon", "changes"],
      properties: {
        zone_id: { type: "string" },
        horizon: { type: "integer", minimum: 1, maximum: 720 },
        changes: {
          type: "object",
          additionalProperties: false,
          properties: {
            capacity_delta: { type: "integer" },
            open_room: { type: "boolean" },
            push_time_minutes: { type: "integer", minimum: 0 },
            staff_delta: { type: "integer" },
          },
        },
      },
    },
    generate_executive_brief: {
      type: "object",
      additionalProperties: false,
      required: ["zone_id", "horizon"],
      properties: {
        zone_id: { type: "string" },
        horizon: { type: "integer", minimum: 1, maximum: 720 },
      },
    },
    generate_professor_brief: {
      type: "object",
      additionalProperties: false,
      required: ["zone_id", "horizon"],
      properties: {
        zone_id: { type: "string" },
        horizon: { type: "integer", minimum: 1, maximum: 720 },
      },
    },
  };

  const validators = Object.fromEntries(
    Object.entries(inputSchemas).map(([name, schema]) => [name, ajv.compile(schema)]),
  );

  async function generateBrief(zoneId, horizon, audience = "executive") {
    const result = await api.post("/api/v1/explain/narrative", {
      zone_id: zoneId,
      horizon,
      audience,
      query: audience === "professor" ? "generate_professor_brief" : "generate_executive_brief",
      language: "de",
      response_mode: "free",
    });

    const response = result?.response || {};
    const context = result?.context || {};

    ensure(validateLlmResponse, response, "llm response output");
    if (validateLlmContextV2) {
      ensure(validateLlmContextV2, context, "llm explainability context v2");
    } else {
      ensure(validateLlmContext, context, "llm explainability context");
    }

    return {
      mode: String(result?.mode || "template"),
      brief: String(result?.narrative_markdown || formatDualOutput(response)),
      response,
      context,
      meta: result?.meta || {},
      warnings: Array.isArray(result?.warnings) ? result.warnings : [],
    };
  }

  const tools = {
    get_live_occupancy: {
      description: "Get latest occupancy point for a zone.",
      inputSchema: inputSchemas.get_live_occupancy,
      run: async (args) => {
        ensure(validators.get_live_occupancy, args, "get_live_occupancy input");
        const now = new Date();
        const from = new Date(now.getTime() - 30 * 60 * 1000).toISOString();
        const history = await api.get("/api/v1/counts", {
          zone_id: args.zone_id,
          from,
          to: now.toISOString(),
          granularity: "raw",
        });
        const points = history.points || [];
        const latest = points.at(-1) || null;
        if (latest) ensure(validateCountPoint, latest, "get_live_occupancy output");
        return { zone_id: args.zone_id, latest };
      },
    },
    get_history: {
      description: "Get historical occupancy points for a zone and time range.",
      inputSchema: inputSchemas.get_history,
      run: async (args) => {
        ensure(validators.get_history, args, "get_history input");
        const result = await api.get("/api/v1/counts", args);
        for (const point of result.points || []) ensure(validateCountPoint, point, "get_history output point");
        return result;
      },
    },
    get_forecast: {
      description: "Get occupancy forecast for a zone and horizon.",
      inputSchema: inputSchemas.get_forecast,
      run: async (args) => {
        ensure(validators.get_forecast, args, "get_forecast input");
        const result = await api.get("/api/v1/forecast", { zone_id: args.zone_id, horizon: args.horizon });
        ensure(validateForecast, result, "get_forecast output");
        return result;
      },
    },
    explain_forecast: {
      description: "Get explainability drivers for a forecast.",
      inputSchema: inputSchemas.explain_forecast,
      run: async (args) => {
        ensure(validators.explain_forecast, args, "explain_forecast input");
        const result = await api.get("/api/v1/explain", { zone_id: args.zone_id, horizon: args.horizon });
        ensure(validateExplanation, result, "explain_forecast output");
        return result;
      },
    },
    list_calendar_events: {
      description: "List calendar events for a zone and range.",
      inputSchema: inputSchemas.list_calendar_events,
      run: async (args) => {
        ensure(validators.list_calendar_events, args, "list_calendar_events input");
        const result = await api.get("/api/v1/calendar/events", args);
        ensure(validateCalendarEvent, result, "list_calendar_events output");
        return result;
      },
    },
    recommend_actions: {
      description: "Get rule-based recommendations for a zone.",
      inputSchema: inputSchemas.recommend_actions,
      run: async (args) => {
        ensure(validators.recommend_actions, args, "recommend_actions input");
        const result = await api.get("/api/v1/recommendations", { zone_id: args.zone_id, horizon: args.horizon });
        ensure(validateRecommendation, result, "recommend_actions output");
        return result;
      },
    },
    simulate_scenario: {
      description: "Simulate a scenario counterfactual (always persist=false in MCP).",
      inputSchema: inputSchemas.simulate_scenario,
      run: async (args) => {
        ensure(validators.simulate_scenario, args, "simulate_scenario input");
        const payload = {
          zone_id: args.zone_id,
          horizon: args.horizon,
          changes: args.changes,
          persist: false,
        };
        const result = await api.post("/api/v1/scenarios/simulate", payload);
        ensure(validateScenarioResult, result, "simulate_scenario output");
        return { ...result, persist: false };
      },
    },
    generate_executive_brief: {
      description: "Generate concise executive brief from forecast/explanation/recommendations.",
      inputSchema: inputSchemas.generate_executive_brief,
      run: async (args) => {
        ensure(validators.generate_executive_brief, args, "generate_executive_brief input");
        return generateBrief(args.zone_id, args.horizon, "executive");
      },
    },
    generate_professor_brief: {
      description: "Generate professor-friendly brief for occupancy understanding and improvements.",
      inputSchema: inputSchemas.generate_professor_brief,
      run: async (args) => {
        ensure(validators.generate_professor_brief, args, "generate_professor_brief input");
        return generateBrief(args.zone_id, args.horizon, "professor");
      },
    },
  };

  return {
    tools,
    async callTool(name, args = {}) {
      const tool = tools[name];
      if (!tool) throw makeError(`unknown tool: ${name}`);
      return tool.run(args);
    },
  };
}
