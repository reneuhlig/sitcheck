import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { App } from "../src/App";

vi.mock("../src/api", () => ({
  fetchHealth: vi.fn(async () => ({ status: "ok", service: "api-gateway" })),
  fetchCommandCenter: vi.fn(async () => ({
    meta: { zone_id: "default-zone", horizon: 15, generated_at: "2026-04-21T10:00:00Z" },
    live: { occupancy: 8, utilization: 0.1, quality_score: 0.85, quality_flags: ["NO_TRACKS", "TRACK_REUSE"], point_count: 20 },
    history: { points: [] },
    forecast_latest: {
      horizon: 15,
      model_version: "lgbm-quantile-v1-20260422153846",
      stale: false,
      points: [{ timestamp: "2026-04-21T11:00:00Z", yhat: 9, pi_low: 8, pi_high: 10 }],
      evidence: { model: { backend: "lgbm" }, quality: { score: 0.85, flags: ["OK"] } },
    },
    weekly_forecast: { points: [] },
    explanation: { drivers: [{ name: "weekday", impact: 0.4 }], uncertainty: { level: "low" } },
    recommendations: {
      actions: [
        { action_type: "staffing", priority: 1, rationale: "Nicht prominent." },
        { action_type: "quality-hardening", priority: 1, rationale: "Daten pruefen." },
      ],
    },
    alerts: [{ code: "ALL_CLEAR", level: "ok", message: "ok" }],
    service_health: [{ service: "forecast", status: "ok" }],
    calendar_events: [],
  })),
  fetchExecutiveNarrative: vi.fn(async () => ({
    mode: "ollama_hybrid",
    meta: { model: "qwen2.5:0.5b", prompt_version: "2.2.0" },
    response: {
      narrative: {
        one_liner: "H15-LGBM ist aktiv und fuer Executive-Entscheidungen verwendbar.",
        empfehlung: "Monitoring fortsetzen.",
      },
      structured: {
        verdict: "go",
        top_drivers: [{ name: "quality", impact: 0.7 }],
        recommended_actions: [{ action_type: "monitor", priority: 1, rationale: "Monitoring fortsetzen." }],
      },
    },
  })),
}));

function renderApp() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>,
  );
}

describe("App", () => {
  it("shows H15 LGBM, Qwen and decision-ready executive state", async () => {
    renderApp();
    expect(await screen.findByText("Entscheidbar")).toBeInTheDocument();
    expect(await screen.findByText("Forecast LGBM")).toBeInTheDocument();
    expect(await screen.findByText("Qwen qwen2.5:0.5b")).toBeInTheDocument();
    expect(await screen.findByText("ALL_CLEAR")).toBeInTheDocument();
    expect((await screen.findAllByText("Monitoring fortsetzen.")).length).toBeGreaterThan(0);
  });
});
