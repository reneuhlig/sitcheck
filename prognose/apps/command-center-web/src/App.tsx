import { useMemo, useState, useCallback, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Clock3,
  DatabaseZap,
  LineChart,
  RefreshCw,
  ShieldAlert,
  Sparkles,
} from "lucide-react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchAudienceNarrative, fetchCommandCenter, fetchExecutiveNarrative, fetchForecastMultiStep, fetchHealth, fetchLive } from "./api";
import { DEFAULT_HORIZON, DEFAULT_ZONE_ID } from "./config";
import { deriveDecisionState, executiveActions, qualityFlags, type StatusTone } from "./decision";
import { compactDate, formatNumber, formatTime, percent } from "./format";
import type {
  AlertItem,
  CalendarEvent,
  CommandCenterPayload,
  DashboardFilters,
  Driver,
  ForecastPoint,
  HistoryPoint,
  NarrativeResponse,
  ServiceHealth,
} from "./types";

const HORIZON_OPTIONS = [210];

function toneIcon(tone: StatusTone) {
  if (tone === "ok") return <CheckCircle2 size={18} />;
  if (tone === "risk") return <ShieldAlert size={18} />;
  if (tone === "warn") return <AlertTriangle size={18} />;
  return <Activity size={18} />;
}

function StatusPill({ label, tone = "info" }: { label: string; tone?: StatusTone }) {
  return <span className={`status-pill status-${tone}`}>{label}</span>;
}

function Panel({
  title,
  icon,
  children,
  className = "",
}: {
  title: string;
  icon?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel-header">
        <div className="panel-title">
          {icon}
          <h2>{title}</h2>
        </div>
      </div>
      {children}
    </section>
  );
}

function DecisionTile({
  label,
  value,
  detail,
  tone = "info",
}: {
  label: string;
  value: string;
  detail: string;
  tone?: StatusTone;
}) {
  return (
    <div className={`decision-tile tile-${tone}`}>
      <div className="tile-label">{label}</div>
      <div className="tile-value">
        {toneIcon(tone)}
        <span>{value}</span>
      </div>
      <div className="tile-detail">{detail}</div>
    </div>
  );
}

type AggInterval = 15 | 30 | 60;

function aggregateForecastPoints(
  points: ForecastPoint[],
  intervalMinutes: AggInterval,
): ForecastPoint[] {
  if (intervalMinutes === 15) return points;
  const step = intervalMinutes / 15;
  return points.filter((_, i) => i % step === step - 1);
}

function buildChartData(
  history: HistoryPoint[] = [],
  forecast: ForecastPoint[] = [],
  aggInterval: AggInterval = 15,
) {
  const historyRows = history.slice(-120).map((point) => ({
    timestamp: point.timestamp,
    label: compactDate(point.timestamp),
    actual: point.occupancy ?? null,
    yhat: null,
    pi_low: null,
    pi_high: null,
  }));
  const aggregated = aggregateForecastPoints(forecast, aggInterval);
  const forecastRows = aggregated.map((point) => ({
    timestamp: point.timestamp,
    label: compactDate(point.timestamp),
    actual: null,
    yhat: point.yhat,
    pi_low: point.pi_low ?? null,
    pi_high: point.pi_high ?? null,
  }));
  return [...historyRows, ...forecastRows];
}

function ForecastChart({
  payload,
  multiPoints,
}: {
  payload: CommandCenterPayload;
  multiPoints?: ForecastPoint[];
}) {
  const [aggInterval, setAggInterval] = useState<AggInterval>(15);

  // Memoize so forecastPoints only changes when underlying data changes, not on every render.
  const forecastPoints = useMemo<ForecastPoint[]>(() => {
    if (multiPoints && multiPoints.length > 0) return multiPoints;
    const multi = payload.forecast_multi?.points ?? [];
    if (multi.length > 0) return multi as ForecastPoint[];
    return (payload.forecast_latest?.points ?? []) as ForecastPoint[];
  }, [multiPoints, payload.forecast_multi, payload.forecast_latest]);

  const forecast = payload.forecast_latest;

  // History rows are stable: only change when backend sends new live data.
  const historyRows = useMemo(
    () =>
      (payload.history?.points || []).slice(-120).map((point) => ({
        label: compactDate(point.timestamp),
        actual: point.occupancy ?? null,
      })),
    [payload.history?.points],
  );

  // Forecast rows change with aggInterval – rebuilds only the forecast part.
  const forecastRows = useMemo(
    () =>
      aggregateForecastPoints(forecastPoints, aggInterval).map((point) => ({
        label: compactDate(point.timestamp),
        yhat: point.yhat,
        pi_low: point.pi_low ?? null,
        pi_high: point.pi_high ?? null,
      })),
    [forecastPoints, aggInterval],
  );

  // Combined only for XAxis so both time ranges are visible.
  const chartData = useMemo(
    () => [
      ...historyRows,
      ...forecastRows.map((r) => ({ label: r.label })),
    ],
    [historyRows, forecastRows],
  );

  const handleAgg = useCallback((interval: AggInterval) => setAggInterval(interval), []);

  const flags = qualityFlags(forecast);
  const firstPoint = forecastPoints[0];
  const lastPoint = forecastPoints[forecastPoints.length - 1];
  const modelVersion =
    payload.forecast_multi?.model_version ?? forecast?.model_version;

  return (
    <Panel title="3h Forecast Verlauf" icon={<LineChart size={19} />} className="forecast-panel">
      <div className="chart-summary">
        <div>
          <span>Live</span>
          <strong>{payload.live?.occupancy ?? "n/a"}</strong>
        </div>
        <div>
          <span>+15 min</span>
          <strong>{formatNumber(firstPoint?.yhat, 1)}</strong>
        </div>
        <div>
          <span>+180 min</span>
          <strong>{formatNumber(lastPoint?.yhat, 1)}</strong>
        </div>
        <div>
          <span>PI 15 min</span>
          <strong>
            {formatNumber(firstPoint?.pi_low, 1)} – {formatNumber(firstPoint?.pi_high, 1)}
          </strong>
        </div>
      </div>
      <div className="agg-buttons">
        {([15, 30, 60] as AggInterval[]).map((interval) => (
          <button
            key={interval}
            type="button"
            className={`agg-btn${aggInterval === interval ? " agg-btn--active" : ""}`}
            onClick={() => handleAgg(interval)}
          >
            {interval} min
          </button>
        ))}
        <span className="agg-label">{forecastPoints.length} Punkte / {forecastRows.length} angezeigt</span>
      </div>
      <div className="chart-frame" data-testid="forecast-chart">
        <ResponsiveContainer width="100%" height={330}>
          <ComposedChart data={chartData} margin={{ top: 12, right: 18, left: -10, bottom: 8 }}>
            <CartesianGrid stroke="#e5e7eb" vertical={false} />
            <XAxis dataKey="label" minTickGap={30} tickLine={false} axisLine={false} />
            <YAxis tickLine={false} axisLine={false} width={42} />
            <Tooltip
              contentStyle={{
                border: "1px solid #d9e0ea",
                borderRadius: 8,
                boxShadow: "0 14px 32px rgba(15, 23, 42, 0.12)",
              }}
            />
            {/* PI band – uses forecastRows so it only rebuilds on agg change, not on history update */}
            <Area
              data={forecastRows}
              type="monotone"
              dataKey="pi_high"
              stroke="none"
              fill="#dbeafe"
              fillOpacity={0.7}
              isAnimationActive={false}
            />
            <Area
              data={forecastRows}
              type="monotone"
              dataKey="pi_low"
              stroke="none"
              fill="#ffffff"
              fillOpacity={1}
              isAnimationActive={false}
            />
            {/* History line – uses historyRows so it stays stable when agg changes */}
            <Line
              data={historyRows}
              type="monotone"
              dataKey="actual"
              stroke="#334155"
              strokeWidth={2}
              dot={false}
              name="History"
              isAnimationActive={false}
            />
            <Line
              data={forecastRows}
              type="monotone"
              dataKey="yhat"
              stroke="#2563eb"
              strokeWidth={3}
              dot={{ r: aggInterval === 60 ? 5 : aggInterval === 30 ? 4 : 3 }}
              name="Forecast"
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="meta-row">
        <StatusPill label={modelVersion || "Modell unbekannt"} tone="info" />
        <StatusPill
          label={forecast?.stale ? "Snapshot stale" : "Snapshot aktuell"}
          tone={forecast?.stale ? "risk" : "ok"}
        />
        {flags.slice(0, 3).map((flag) => (
          <StatusPill key={flag} label={flag} tone="warn" />
        ))}
      </div>
    </Panel>
  );
}

function narrativeText(narrative?: NarrativeResponse): string {
  const block = narrative?.response?.narrative;
  if (!block) {
    return "";
  }
  const text =
    block.narrative_markdown ||
    block.markdown ||
    block.executive_summary ||
    block.body ||
    (Array.isArray(block.key_points) ? block.key_points.join("\n") : "") ||
    [block.one_liner, block.warum, block.unsicherheit, block.empfehlung, block.evidence_hinweis]
      .filter((value) => typeof value === "string" && value.trim())
      .join("\n\n");
  return String(text || "").trim();
}

function TextBlock({ text }: { text: string }) {
  return (
    <div className="text-block">
      {text.split(/\n{2,}|\n-/).map((part, index) => {
        const cleaned = part.replace(/^\s*[-*]\s*/, "").trim();
        if (!cleaned) return null;
        return <p key={`${cleaned.slice(0, 12)}-${index}`}>{cleaned}</p>;
      })}
    </div>
  );
}

function ExecutiveBrief({
  narrative,
  error,
  isLoading,
  degraded,
}: {
  narrative?: NarrativeResponse;
  error?: Error | null;
  isLoading: boolean;
  degraded: boolean;
}) {
  const text = narrativeText(narrative);
  return (
    <Panel title="Executive Brief" icon={<Sparkles size={19} />} className="brief-panel">
      {degraded ? (
        <div className="degrade-callout">
          <ShieldAlert size={19} />
          <span>Prognose ist degradiert und nicht normal belastbar.</span>
        </div>
      ) : null}
      {isLoading ? <div className="skeleton">Qwen erstellt den Executive Brief...</div> : null}
      {error ? (
        <div className="error-box" data-testid="qwen-error">
          Qwen/Ollama hat keine valide Antwort geliefert: {error.message}
        </div>
      ) : null}
      {!isLoading && !error && text ? <TextBlock text={text} /> : null}
      {!isLoading && !error && !text ? <div className="empty-state">Kein Qwen-Brief vorhanden.</div> : null}
      <div className="technical-strip">
        <span>mode={narrative?.mode || "pending"}</span>
        <span>model={narrative?.meta?.model || "qwen2.5:0.5b"}</span>
        <span>prompt={narrative?.meta?.prompt_version || "-"}</span>
      </div>
    </Panel>
  );
}

function DriverList({ drivers = [] }: { drivers?: Driver[] }) {
  const top = drivers.slice(0, 5);
  if (!top.length) {
    return <div className="empty-state">Keine Treiber verfuegbar.</div>;
  }
  const max = Math.max(...top.map((driver) => Math.abs(Number(driver.impact || 0))), 1);
  return (
    <div className="driver-list">
      {top.map((driver, index) => {
        const impact = Number(driver.impact || 0);
        return (
          <div className="driver-row" key={`${driver.name || "driver"}-${index}`}>
            <div className="driver-name">{driver.name || "Treiber"}</div>
            <div className="driver-bar">
              <span style={{ width: `${Math.max(8, (Math.abs(impact) / max) * 100)}%` }} />
            </div>
            <div className="driver-impact">{formatNumber(impact, 2)}</div>
          </div>
        );
      })}
    </div>
  );
}

function ActionList({ payload, narrative }: { payload: CommandCenterPayload; narrative?: NarrativeResponse }) {
  const actions = executiveActions(payload, narrative);
  if (!actions.length) {
    return <div className="empty-state">Keine sichere Executive-Aktion verfuegbar.</div>;
  }
  return (
    <div className="action-list">
      {actions.map((action, index) => (
        <article className="action-item" key={`${action.action_type || "action"}-${index}`}>
          <div>
            <strong>{action.action_type || "monitor"}</strong>
            <span>Prioritaet {action.priority || "n/a"}</span>
          </div>
          <p>{action.rationale || "Lage beobachten und Statuswechsel bewerten."}</p>
        </article>
      ))}
    </div>
  );
}

function WeeklyRisk({ payload }: { payload: CommandCenterPayload }) {
  const points = payload.weekly_forecast?.points || [];
  const peak = points.reduce((max, point) => Math.max(max, point.yhat || 0), 0);
  const avg = points.length ? points.reduce((sum, point) => sum + (point.yhat || 0), 0) / points.length : 0;
  const interval = points.length
    ? points.reduce((sum, point) => sum + ((point.pi_high || point.yhat || 0) - (point.pi_low || point.yhat || 0)), 0) /
      points.length
    : 0;
  return (
    <div className="weekly-grid">
      <div>
        <span>Peak</span>
        <strong>{formatNumber(peak, 1)}</strong>
      </div>
      <div>
        <span>Durchschnitt</span>
        <strong>{formatNumber(avg, 1)}</strong>
      </div>
      <div>
        <span>Unsicherheit</span>
        <strong>{formatNumber(interval, 1)}</strong>
      </div>
    </div>
  );
}

function EventsList({ events = [] }: { events?: CalendarEvent[] }) {
  const top = events.slice(0, 5);
  if (!top.length) {
    return <div className="empty-state">Keine Kalenderereignisse im Fenster.</div>;
  }
  return (
    <div className="events-list">
      {top.map((event, index) => (
        <div className="event-row" key={`${event.title || "event"}-${index}`}>
          <div>
            <strong>{event.title || "Kalenderereignis"}</strong>
            <span>{event.category || event.source || "context"}</span>
          </div>
          <time>{formatTime(event.starts_at)}</time>
        </div>
      ))}
    </div>
  );
}

function ServiceGrid({ services = [] }: { services?: ServiceHealth[] }) {
  if (!services.length) {
    return <div className="empty-state">Keine Service-Health verfuegbar.</div>;
  }
  return (
    <div className="service-grid">
      {services.map((service) => {
        const ok = service.status === "ok";
        return (
          <div className="service-item" key={service.service}>
            <span className={ok ? "dot dot-ok" : "dot dot-risk"} />
            <strong>{service.service}</strong>
            <span>{service.status}</span>
          </div>
        );
      })}
    </div>
  );
}

function Alerts({ alerts = [] }: { alerts?: AlertItem[] }) {
  if (!alerts.length) {
    return null;
  }
  return (
    <div className="alerts-row">
      {alerts.map((alert) => (
        <StatusPill
          key={`${alert.code}-${alert.message}`}
          label={alert.code}
          tone={alert.level === "risk" ? "risk" : alert.level === "warn" ? "warn" : alert.level === "ok" ? "ok" : "info"}
        />
      ))}
    </div>
  );
}

const ROLE_CONFIGS: { label: string; audience: string; template: string }[] = [
  {
    label: "OPS",
    audience: "ops",
    template:
      "Gibt es aktuell Service-Probleme oder Qualitätsauffälligkeiten? Fasse zusammen was sofort geprüft werden muss.",
  },
  {
    label: "Executive",
    audience: "executive",
    template: "Schreibe einen Executive Brief: Aktuelle Lage, Risiko und eine klare Handlungsempfehlung.",
  },
  {
    label: "Enduser",
    audience: "enduser",
    template:
      "Ist jetzt ein guter Zeitpunkt zum Lernen in der Bibliothek? Wie entwickelt sich die Auslastung in den nächsten Minuten?",
  },
  {
    label: "Professor",
    audience: "professor",
    template:
      "Erkläre die wichtigsten Prognosetreiber und die Modellqualität. Welche Features dominieren und wie valide ist die Unsicherheitsschätzung?",
  },
  {
    label: "Freie Frage",
    audience: "auto",
    template: "",
  },
];

function AudienceChat({ filters }: { filters: DashboardFilters }) {
  const [activeTab, setActiveTab] = useState(0);
  const [queries, setQueries] = useState<Record<number, string>>(
    Object.fromEntries(ROLE_CONFIGS.map((r, i) => [i, r.template])),
  );
  const [responses, setResponses] = useState<Record<number, string>>({});
  const [loading, setLoading] = useState<Record<number, boolean>>({});
  const [errors, setErrors] = useState<Record<number, string>>({});

  const currentConfig = ROLE_CONFIGS[activeTab];

  async function handleSend() {
    const query = (queries[activeTab] ?? currentConfig.template).trim();
    if (!query) return;
    setLoading((prev) => ({ ...prev, [activeTab]: true }));
    setErrors((prev) => ({ ...prev, [activeTab]: "" }));
    try {
      const result = await fetchAudienceNarrative(query, currentConfig.audience, filters);
      const text = narrativeText(result);
      setResponses((prev) => ({ ...prev, [activeTab]: text || JSON.stringify(result.response, null, 2) }));
    } catch (error) {
      setErrors((prev) => ({
        ...prev,
        [activeTab]: error instanceof Error ? error.message : String(error),
      }));
    } finally {
      setLoading((prev) => ({ ...prev, [activeTab]: false }));
    }
  }

  const currentQuery = queries[activeTab] ?? currentConfig.template;
  const isBusy = Boolean(loading[activeTab]);

  return (
    <Panel title="Frage an den Assistenten" icon={<Sparkles size={19} />} className="chat-panel">
      <div className="chat-tabs" role="tablist">
        {ROLE_CONFIGS.map((config, index) => (
          <button
            key={config.label}
            role="tab"
            aria-selected={activeTab === index}
            className={`chat-tab${activeTab === index ? " chat-tab--active" : ""}`}
            onClick={() => setActiveTab(index)}
            type="button"
          >
            {config.label}
          </button>
        ))}
      </div>
      <div className="chat-body">
        <textarea
          className="chat-textarea"
          value={currentQuery}
          onChange={(e) => setQueries((prev) => ({ ...prev, [activeTab]: e.target.value }))}
          placeholder="Deine Frage an das LLM…"
          rows={4}
        />
        <button
          className="send-button"
          type="button"
          onClick={() => void handleSend()}
          disabled={isBusy || !currentQuery.trim()}
        >
          {isBusy ? "Analysiere…" : "Senden"}
        </button>
        {errors[activeTab] ? <div className="error-box">{errors[activeTab]}</div> : null}
        {responses[activeTab] ? (
          <div className="chat-response">
            <TextBlock text={responses[activeTab]} />
          </div>
        ) : null}
      </div>
    </Panel>
  );
}

export function App() {
  const [zoneId, setZoneId] = useState(DEFAULT_ZONE_ID);
  const [horizon, setHorizon] = useState(DEFAULT_HORIZON);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const filters: DashboardFilters = {
    zoneId,
    horizon,
    historyMinutes: 180,
    staleSeconds: 900,
    longTermDays: 14,
  };

  // ── Independent queries – each fails on its own without blocking others ──

  // 1. API health (lightweight)
  const healthQuery = useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    refetchInterval: autoRefresh ? 30_000 : false,
  });

  // 2. Live occupancy + history – DB only, always fast, never blocked by ML services
  const liveQuery = useQuery({
    queryKey: ["live", zoneId],
    queryFn: () => fetchLive(zoneId, 180),
    refetchInterval: autoRefresh ? 15_000 : false,
    staleTime: 10_000,
  });

  // 3. Full aggregated data – may call forecast/xai/recommendations services
  //    Backend now returns 200 even when sub-services are down.
  const commandQuery = useQuery({
    queryKey: ["command-center", filters],
    queryFn: () => fetchCommandCenter(filters),
    refetchInterval: autoRefresh ? 30_000 : false,
  });

  // 4. Multi-step ML forecast – cached per 15-min slot on backend
  const FORECAST_MULTI_INTERVAL = 13 * 60_000;
  const forecastMultiQuery = useQuery({
    queryKey: ["forecast-multi", zoneId],
    queryFn: () => fetchForecastMultiStep(zoneId),
    staleTime: FORECAST_MULTI_INTERVAL,
    refetchInterval: autoRefresh ? FORECAST_MULTI_INTERVAL : false,
  });

  // 5. Narrative (Qwen) – cached per 15-min slot on backend
  const NARRATIVE_INTERVAL = 13 * 60_000;
  const narrativeQuery = useQuery({
    queryKey: ["executive-narrative", zoneId, horizon],
    queryFn: () => fetchExecutiveNarrative(filters),
    enabled: Boolean(commandQuery.data || liveQuery.data),
    retry: false,
    staleTime: NARRATIVE_INTERVAL,
    refetchInterval: autoRefresh ? NARRATIVE_INTERVAL : false,
  });

  // Merge live data: prefer liveQuery (fresher, 15s cadence) over commandQuery
  const liveOverride = liveQuery.data;
  const payload = commandQuery.data;
  const mergedPayload: CommandCenterPayload | undefined = payload
    ? {
        ...payload,
        live: liveOverride?.live ?? payload.live,
        history: liveOverride?.history ?? payload.history,
      }
    : liveOverride
      ? { live: liveOverride.live, history: liveOverride.history, meta: { zone_id: zoneId, generated_at: liveOverride.generated_at } }
      : undefined;

  const decision = mergedPayload ? deriveDecisionState(mergedPayload as CommandCenterPayload, narrativeQuery.data) : null;
  const forecastModel = payload?.forecast_latest?.model_version || "unbekannt";
  const forecastIsLgbm = forecastModel.toLowerCase().includes("lgbm");

  function refreshAll() {
    void healthQuery.refetch();
    void liveQuery.refetch();
    void commandQuery.refetch();
    void narrativeQuery.refetch();
    void forecastMultiQuery.refetch();
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark">
            <BarChart3 size={22} />
          </div>
          <div>
            <h1>Sitcheck Command Center</h1>
            <span>Executive Decision Cockpit</span>
          </div>
        </div>
        <div className="control-strip">
          <label>
            Zone
            <input value={zoneId} onChange={(event) => setZoneId(event.target.value)} />
          </label>
          <label>
            Horizont
            <select value={horizon} onChange={(event) => setHorizon(Number(event.target.value))}>
              {HORIZON_OPTIONS.map((option) => (
                <option value={option} key={option}>
                  {option} min
                </option>
              ))}
            </select>
          </label>
          <button className="toggle-button" type="button" onClick={() => setAutoRefresh((value) => !value)}>
            <Clock3 size={16} />
            {autoRefresh ? "Auto" : "Manuell"}
          </button>
          <button className="icon-button" type="button" onClick={refreshAll} aria-label="Refresh">
            <RefreshCw size={17} />
          </button>
        </div>
      </header>

      <section className="status-bar">
        <StatusPill
          label={`API ${healthQuery.data?.status || (healthQuery.isError ? "down" : "prueft")}`}
          tone={healthQuery.isError ? "risk" : healthQuery.data?.status === "ok" ? "ok" : "info"}
        />
        <StatusPill
          label={liveQuery.isError ? "Live: Fehler" : liveQuery.data ? "Live: ok" : "Live: lädt"}
          tone={liveQuery.isError ? "risk" : liveQuery.data ? "ok" : "info"}
        />
        <StatusPill label={forecastIsLgbm ? "Forecast LGBM" : `Forecast ${forecastModel}`} tone={forecastIsLgbm ? "ok" : "risk"} />
        <StatusPill
          label={`Qwen ${narrativeQuery.data?.meta?.model || (narrativeQuery.isError ? "Fehler" : "prueft")}`}
          tone={narrativeQuery.isError ? "risk" : narrativeQuery.data?.meta?.model ? "ok" : "info"}
        />
        <span className="last-refresh">
          Live {formatTime(liveOverride?.generated_at)} · CC {formatTime(payload?.meta?.generated_at)}
        </span>
      </section>

      {/* Live data loads immediately – no blocking on command-center */}
      {liveQuery.isLoading && !mergedPayload ? (
        <div className="loading-page">Echtzeit-Daten werden geladen...</div>
      ) : null}

      {/* Command-center error is non-fatal – show warning but keep live data visible */}
      {commandQuery.isError ? (
        <div className="error-box" style={{ margin: "0.5rem 1rem" }}>
          Aggregierte Daten nicht verfügbar: {(commandQuery.error as Error).message}
        </div>
      ) : null}

      {mergedPayload ? (
        <>
          {decision ? (
            <section className="decision-band" data-testid="decision-band">
              <DecisionTile label="Entscheidbarkeit" value={decision.label} detail={decision.explanation} tone={decision.tone} />
              <DecisionTile
                label="Forecast-Modell"
                value={decision.modelVersion}
                detail={forecastIsLgbm ? "Primaerer LGBM-Pfad aktiv" : "Nicht primaerer Modellpfad"}
                tone={decision.forecastTone}
              />
              <DecisionTile
                label="Datenqualitaet"
                value={decision.dataQualityLabel}
                detail={decision.reasons.slice(0, 2).join(" | ") || "Keine blockierenden Flags"}
                tone={decision.degraded ? "risk" : "ok"}
              />
              <DecisionTile
                label="Naechste Aktion"
                value={decision.primaryAction.label}
                detail={decision.primaryAction.detail}
                tone={decision.degraded ? "warn" : "ok"}
              />
            </section>
          ) : null}

          <Alerts alerts={(mergedPayload as CommandCenterPayload).alerts} />

          <section className="main-grid">
            <ForecastChart payload={mergedPayload as CommandCenterPayload} multiPoints={forecastMultiQuery.data?.points} />
            <ExecutiveBrief
              narrative={narrativeQuery.data}
              error={narrativeQuery.error as Error | null}
              isLoading={narrativeQuery.isLoading}
              degraded={decision?.degraded ?? false}
            />
          </section>

          {payload ? (
            <>
              <section className="detail-grid">
                <Panel title="Top-Treiber" icon={<DatabaseZap size={19} />}>
                  <DriverList drivers={narrativeQuery.data?.response?.structured?.top_drivers || payload.explanation?.drivers} />
                </Panel>
                <Panel title="Empfehlungen" icon={<CheckCircle2 size={19} />}>
                  <ActionList payload={payload} narrative={narrativeQuery.data} />
                </Panel>
                <Panel title="Wochenrisiko" icon={<Activity size={19} />}>
                  <WeeklyRisk payload={payload} />
                </Panel>
                <Panel title="Kalenderkontext" icon={<Clock3 size={19} />}>
                  <EventsList events={payload.calendar_events} />
                </Panel>
              </section>

              <section className="technical-grid">
                <details>
                  <summary>Systemstatus</summary>
                  <ServiceGrid services={payload.service_health} />
                </details>
                <details>
                  <summary>Evidenz und Lineage</summary>
                  <pre>{JSON.stringify({ forecast: payload.forecast_latest?.evidence, lineage: payload.model_lineage }, null, 2)}</pre>
                </details>
                <details>
                  <summary>Qwen Details</summary>
                  <pre>{JSON.stringify(narrativeQuery.data || { error: narrativeQuery.error?.message }, null, 2)}</pre>
                </details>
              </section>

              <AudienceChat filters={filters} />
            </>
          ) : (
            <div className="loading-page" style={{ marginTop: "1rem" }}>
              Forecast & Analyse laden… (Live-Daten bereits verfügbar)
            </div>
          )}
        </>
      ) : null}
    </main>
  );
}
