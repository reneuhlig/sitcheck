import crypto from "node:crypto";

const SYSTEM_PROMPT_DE = [
  "Du bist Sitcheck Explainability Assistant.",
  "Nutze ausschließlich die gelieferten ECP-Daten.",
  "Wenn Evidenz fehlt, sage das explizit.",
  "Erfinde keine Ursachen, Quellen oder Zahlen.",
  "Jede Kernaussage braucht evidence_refs.",
  "Ausgabeformat: JSON-Objekt mit narrative und structured.",
].join(" ");

const AUDIENCE_PROMPTS_DE = {
  ops: "Zielgruppe: Betriebsteam. Fokus: kurzfristige Peaks, Quality Flags, konkrete nächste Schritte.",
  executive: "Zielgruppe: Management. Fokus: Risiko, Auswirkung, Entscheidungsempfehlung, Evidenz-ID.",
  enduser: "Zielgruppe: nicht-technische Nutzer. Fokus: einfache Sprache und Sicherheitseinordnung.",
};

const GUARDRAILS = [
  "no_free_sql",
  "api_tools_only",
  "no_invented_facts",
  "each_claim_needs_evidence_ref",
  "no_hard_action_if_uncertainty_high",
];

function nowIso() {
  return new Date().toISOString();
}

function asNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function asInt(value, fallback = 0) {
  const n = Number(value);
  return Number.isInteger(n) ? n : fallback;
}

function pickRef(citationMap, preferredTypes) {
  for (const sourceType of preferredTypes) {
    for (const item of citationMap) {
      if (item?.source_type === sourceType) return String(item.ref_id);
    }
  }
  if (citationMap.length) return String(citationMap[0].ref_id);
  return "ref-missing";
}

export function buildHistoryDigest(historyPayload = {}) {
  const points = Array.isArray(historyPayload?.points) ? historyPayload.points : [];
  if (!points.length) {
    return {
      last_occupancy: 0,
      last_utilization: 0,
      trend_15m: 0,
      trend_60m: 0,
      quality_score_avg: 0.5,
      quality_flags_top: ["NO_HISTORY"],
      point_count: 0,
    };
  }

  const occupancies = points.map((p) => asNumber(p?.occupancy, 0));
  const utilizations = points.map((p) => asNumber(p?.utilization, 0));
  const qualityScores = points.map((p) => asNumber(p?.quality_score, 0.5));

  const mean = (arr) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0);

  const lastOccupancy = occupancies[occupancies.length - 1] ?? 0;
  const lastUtilization = utilizations[utilizations.length - 1] ?? 0;
  const mean15 = mean(occupancies.slice(-15));
  const mean60 = mean(occupancies.slice(-60));

  const flags = new Map();
  for (const point of points) {
    const pointFlags = Array.isArray(point?.quality_flags) ? point.quality_flags : [];
    for (const flag of pointFlags) {
      if (typeof flag !== "string") continue;
      flags.set(flag, (flags.get(flag) || 0) + 1);
    }
  }
  if (!flags.size) flags.set("OK", 1);

  const topFlags = [...flags.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([flag]) => flag);

  return {
    last_occupancy: Number(lastOccupancy.toFixed(3)),
    last_utilization: Number(lastUtilization.toFixed(6)),
    trend_15m: Number((lastOccupancy - mean15).toFixed(3)),
    trend_60m: Number((lastOccupancy - mean60).toFixed(3)),
    quality_score_avg: Math.max(0, Math.min(1, Number(mean(qualityScores).toFixed(6)))),
    quality_flags_top: topFlags,
    point_count: points.length,
  };
}

export function buildCitationMap(...evidenceObjects) {
  const refs = [];
  const seen = new Set();

  for (const evidence of evidenceObjects) {
    if (!evidence || typeof evidence !== "object") continue;

    const evidenceId = String(evidence.evidence_id || "ev-missing");
    const timeWindow = evidence.time_window && typeof evidence.time_window === "object" ? evidence.time_window : {};
    const model = evidence.model && typeof evidence.model === "object" ? evidence.model : {};
    const quality = evidence.quality && typeof evidence.quality === "object" ? evidence.quality : {};
    const sources = Array.isArray(evidence.sources) ? evidence.sources : [];

    for (const source of sources) {
      if (!source || typeof source !== "object") continue;
      const sourceType = String(source.type || "api");
      const sourceId = String(source.id || "unknown");
      const key = `${evidenceId}|${sourceType}|${sourceId}`;
      if (seen.has(key)) continue;
      seen.add(key);

      refs.push({
        ref_id: `ref-${refs.length + 1}`,
        evidence_id: evidenceId,
        source_type: sourceType,
        source_id: sourceId,
        time_window: {
          from: String(timeWindow.from || nowIso()),
          to: String(timeWindow.to || nowIso()),
        },
        model_version: String(model.version || "unknown"),
        quality_score: Math.max(0, Math.min(1, asNumber(quality.score, 0.5))),
        quality_flags: Array.isArray(quality.flags) ? quality.flags.map((x) => String(x)) : [],
      });
    }
  }

  if (!refs.length) {
    refs.push({
      ref_id: "ref-1",
      evidence_id: "ev-missing",
      source_type: "api",
      source_id: "none",
      time_window: { from: nowIso(), to: nowIso() },
      model_version: "unknown",
      quality_score: 0,
      quality_flags: ["NO_EVIDENCE"],
    });
  }

  return refs;
}

export function buildExplainabilityContext({
  forecastLatest,
  explanation,
  recommendation,
  history,
  scenarioDigest = null,
  audience = "executive",
  language = "de",
  timezone = "UTC",
  query = "",
}) {
  return {
    forecast_latest: forecastLatest,
    explanation,
    recommendation,
    history_digest: buildHistoryDigest(history),
    scenario_digest: scenarioDigest,
    context_meta: {
      request_id: `ctx-${crypto.randomUUID()}`,
      generated_at: nowIso(),
      audience,
      language,
      timezone,
      guardrails: GUARDRAILS,
      query,
    },
    citation_map: buildCitationMap(
      forecastLatest?.evidence,
      explanation?.evidence,
      recommendation?.evidence,
      scenarioDigest?.evidence,
    ),
  };
}

function confidenceStatement(level, score) {
  if (level === "high") return `Niedrige Verlässlichkeit (Unsicherheit=${score.toFixed(2)}). Aussagen als Tendenz nutzen.`;
  if (level === "medium") return `Mittlere Verlässlichkeit (Unsicherheit=${score.toFixed(2)}). Monitoring empfohlen.`;
  return `Hohe Verlässlichkeit (Unsicherheit=${score.toFixed(2)}).`;
}

export function renderTemplateFallback(context) {
  const forecast = context?.forecast_latest || {};
  const explain = context?.explanation || {};
  const recommendation = context?.recommendation || {};
  const audience = String(context?.context_meta?.audience || "executive");
  const citationMap = Array.isArray(context?.citation_map) ? context.citation_map : [];

  const zoneId = String(forecast.zone_id || "unknown-zone");
  const horizon = asInt(forecast.horizon, 60);
  const stale = Boolean(forecast.stale);

  const uncertainty = explain?.uncertainty && typeof explain.uncertainty === "object" ? explain.uncertainty : {};
  const uncertaintyLevel = String(uncertainty.level || "high");
  const uncertaintyScore = Math.max(0, Math.min(1, asNumber(uncertainty.score, 0.8)));

  const gates = recommendation?.gates && typeof recommendation.gates === "object" ? recommendation.gates : {};
  const qualityOk = Boolean(gates.quality_ok ?? true);
  const uncertaintyOk = Boolean(gates.uncertainty_ok ?? true);

  let verdict = "monitor";
  if (stale || !qualityOk) verdict = "blocked";
  else if (uncertaintyLevel === "high" || !uncertaintyOk) verdict = "attention";
  else if ((recommendation.actions || []).some((a) => a?.action_type !== "monitor")) verdict = "action_needed";

  const refCounts = pickRef(citationMap, ["counts", "forecast", "api"]);
  const refEvents = pickRef(citationMap, ["events", "counts", "api"]);
  const refXai = pickRef(citationMap, ["xai", "forecast", "counts"]);
  const refRec = pickRef(citationMap, ["recommendation", "forecast", "counts"]);

  const rawDrivers = Array.isArray(explain?.drivers) ? explain.drivers : [];
  const topDrivers = rawDrivers.slice(0, 3).map((driver) => ({
    name: String(driver?.name || "unknown"),
    impact: asNumber(driver?.impact, 0),
    direction: String(driver?.direction || "mixed"),
    evidence_ref: String(driver?.name) === "event_context" ? refEvents : refCounts,
  }));

  let rawActions = Array.isArray(recommendation?.actions) ? recommendation.actions : [];
  if (uncertaintyLevel === "high") {
    rawActions = [{ action_type: "monitor", priority: 1, rationale: "Unsicherheit ist hoch, daher nur überwachen." }];
  }
  if (!rawActions.length) {
    rawActions = [{ action_type: "monitor", priority: 1, rationale: "Keine belastbare Aktion abgeleitet." }];
  }

  const recommendedActions = rawActions.map((action) => ({
    action_type: String(action?.action_type || "monitor"),
    priority: asInt(action?.priority, 1),
    rationale: String(action?.rationale || ""),
    evidence_ref: refRec,
  }));

  const evidenceRefs = [...new Set([
    ...topDrivers.map((x) => x.evidence_ref),
    ...recommendedActions.map((x) => x.evidence_ref),
    refXai,
  ])].filter(Boolean);

  const limitations = [];
  if (stale) limitations.push("Forecast-Snapshot ist veraltet.");
  if (!qualityOk) limitations.push("Datenqualitäts-Gate blockiert harte Empfehlungen.");
  if (uncertaintyLevel === "high") limitations.push("Hohe Unsicherheit reduziert Entscheidungssicherheit.");
  if (!limitations.length) limitations.push("Keine kritischen Einschränkungen erkannt.");

  return {
    narrative: {
      one_liner: `Prognose für Zone ${zoneId}: Urteil = ${verdict}.`,
      warum: `Haupttreiber: ${topDrivers.map((d) => d.name).join(", ") || "keine"}. Basis-Evidenz [ref:${refCounts}].`,
      unsicherheit: `Unsicherheit ist ${uncertaintyLevel} (Score ${uncertaintyScore.toFixed(2)}) [ref:${refXai}].`,
      empfehlung: `Empfohlene Aktion: ${recommendedActions[0].action_type} [ref:${recommendedActions[0].evidence_ref}].`,
      evidence_hinweis: `Evidenzreferenzen: ${evidenceRefs.map((r) => `[ref:${r}]`).join(", ")}`,
    },
    structured: {
      audience,
      zone_id: zoneId,
      horizon,
      verdict,
      top_drivers: topDrivers,
      uncertainty: {
        score: uncertaintyScore,
        level: uncertaintyLevel,
        reason: String(uncertainty.reason || "unspecified"),
        evidence_ref: refXai,
      },
      recommended_actions: recommendedActions,
      evidence_refs: evidenceRefs.length ? evidenceRefs : ["ref-1"],
      confidence_statement: confidenceStatement(uncertaintyLevel, uncertaintyScore),
      limitations,
    },
  };
}

export function buildLlmPrompt(context, audience = "executive") {
  const audiencePrompt = AUDIENCE_PROMPTS_DE[audience] || AUDIENCE_PROMPTS_DE.ops;
  return [
    SYSTEM_PROMPT_DE,
    audiencePrompt,
    "Antworte ausschließlich als JSON mit den Schlüsseln narrative und structured.",
    `Kontext (ECP v1): ${JSON.stringify(context)}`,
  ].join("\n\n");
}

export function extractJsonObject(rawText = "") {
  const start = rawText.indexOf("{");
  if (start < 0) return null;

  let depth = 0;
  let end = -1;
  for (let i = start; i < rawText.length; i += 1) {
    const ch = rawText[i];
    if (ch === "{") depth += 1;
    if (ch === "}") depth -= 1;
    if (depth === 0) {
      end = i;
      break;
    }
  }

  if (end < 0) return null;
  try {
    return JSON.parse(rawText.slice(start, end + 1));
  } catch {
    return null;
  }
}

export function validateDualResponseShape(response) {
  if (!response || typeof response !== "object") return { ok: false, reason: "response is not object" };
  if (!response.narrative || !response.structured) return { ok: false, reason: "missing narrative/structured" };

  const narrativeKeys = ["one_liner", "warum", "unsicherheit", "empfehlung", "evidence_hinweis"];
  for (const key of narrativeKeys) {
    if (!(key in response.narrative)) return { ok: false, reason: `narrative key missing: ${key}` };
  }

  const structuredKeys = [
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
  ];
  for (const key of structuredKeys) {
    if (!(key in response.structured)) return { ok: false, reason: `structured key missing: ${key}` };
  }

  if (!Array.isArray(response.structured.evidence_refs) || response.structured.evidence_refs.length < 1) {
    return { ok: false, reason: "evidence_refs missing" };
  }

  for (const driver of response.structured.top_drivers || []) {
    if (!driver?.evidence_ref) return { ok: false, reason: "driver missing evidence_ref" };
  }
  for (const action of response.structured.recommended_actions || []) {
    if (!action?.evidence_ref) return { ok: false, reason: "action missing evidence_ref" };
  }
  if (!response.structured.uncertainty?.evidence_ref) return { ok: false, reason: "uncertainty missing evidence_ref" };

  return { ok: true, reason: null };
}

export function formatDualOutput(response) {
  const n = response?.narrative || {};
  return [
    `${n.one_liner || ""}`,
    `Warum: ${n.warum || ""}`,
    `Unsicherheit: ${n.unsicherheit || ""}`,
    `Empfehlung: ${n.empfehlung || ""}`,
    `Evidenz: ${n.evidence_hinweis || ""}`,
    "",
    "```json",
    JSON.stringify(response, null, 2),
    "```",
  ].join("\n");
}
