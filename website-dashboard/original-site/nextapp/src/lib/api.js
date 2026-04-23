"use client";

const DEFAULT_ZONE_ID = "default-zone";
const DEFAULT_HORIZON = 210;
const DEFAULT_HISTORY_MINUTES = 180;
const DEFAULT_EXPLAIN_OLLAMA_MODEL = "qwen2.5:0.5b";

function normalizeBaseUrl() {
  const envBaseUrl = process.env.NEXT_PUBLIC_BACKEND_URL?.trim();
  if (!envBaseUrl) {
    return "";
  }
  return envBaseUrl.replace(/\/$/, "");
}

function buildUrl(path) {
  const safePath = path.startsWith("/") ? path : `/${path}`;
  return `${normalizeBaseUrl()}${safePath}`;
}

function extractErrorMessage(payload, fallbackMessage) {
  if (!payload || typeof payload !== "object") {
    return fallbackMessage;
  }
  if (typeof payload.detail === "string" && payload.detail.trim()) {
    return payload.detail;
  }
  if (typeof payload.error === "string" && payload.error.trim()) {
    return payload.error;
  }
  return fallbackMessage;
}

async function apiRequest(path, options = {}) {
  const { method = "GET", body, headers = {}, signal } = options;
  const response = await fetch(buildUrl(path), {
    method,
    credentials: "include",
    headers: {
      ...(body ? { "Content-Type": "application/json" } : {}),
      ...headers,
    },
    body: body ? JSON.stringify(body) : undefined,
    signal,
  });

  const contentType = response.headers.get("content-type") || "";
  const isJson = contentType.includes("application/json");
  const payload = isJson ? await response.json() : await response.text();

  if (!response.ok) {
    throw new Error(extractErrorMessage(payload, `API-Fehler (${response.status})`));
  }

  return payload;
}

export const sitcheckApi = {
  defaultZoneId: DEFAULT_ZONE_ID,
  async getHubOverview(signal) {
    return apiRequest("/api/hub/overview", { signal });
  },
  async getDhbwMannheimContext(signal) {
    return apiRequest("/api/hub/dhbw-mannheim-context", { signal });
  },
  async getZones(signal) {
    return apiRequest("/api/v1/zones", { signal });
  },
  async getCommandCenter(signal) {
    return apiRequest(
      `/api/v1/dashboard/command-center?zone_id=${DEFAULT_ZONE_ID}&horizon=${DEFAULT_HORIZON}&history_minutes=${DEFAULT_HISTORY_MINUTES}`,
      { signal },
    );
  },
  async getExplainNarrative(options = {}) {
    const {
      signal,
      query,
      audience = "enduser",
      language = "de",
      horizon = DEFAULT_HORIZON,
      zoneId = DEFAULT_ZONE_ID,
      ollamaModel = DEFAULT_EXPLAIN_OLLAMA_MODEL,
    } = options;

    return apiRequest("/api/v1/explain/narrative", {
      method: "POST",
      body: {
        zone_id: zoneId,
        horizon,
        audience,
        language,
        query:
          typeof query === "string" && query.trim()
            ? query.trim()
            : "Erkläre in einfacher deutscher Sprache, warum die Bibliothek ab 30 Minuten nach der aktuellen Uhrzeit für die folgenden 3 Stunden so prognostiziert wird.",
        response_mode: "free",
        ollama_model: ollamaModel,
      },
      signal,
    });
  },
  async authMe(signal) {
    return apiRequest("/api/v1/auth/me", { signal });
  },
  async register(credentials) {
    return apiRequest("/api/v1/auth/register", {
      method: "POST",
      body: credentials,
    });
  },
  async login(credentials) {
    return apiRequest("/api/v1/auth/login", {
      method: "POST",
      body: credentials,
    });
  },
  async logout() {
    return apiRequest("/api/v1/auth/logout", { method: "POST" });
  },
  async getCounts({ granularity = "1m", historyMinutes = 180, signal } = {}) {
    const now = new Date();
    const from = new Date(now.getTime() - historyMinutes * 60 * 1000);
    return apiRequest(
      `/api/v1/counts?zone_id=${DEFAULT_ZONE_ID}&from=${encodeURIComponent(from.toISOString())}&to=${encodeURIComponent(now.toISOString())}&granularity=${encodeURIComponent(granularity)}`,
      { signal },
    );
  },
  async getBookings(signal) {
    return apiRequest("/api/v1/bookings", { signal });
  },
  async getAdminBookings(signal) {
    return apiRequest("/api/v1/admin/bookings", { signal });
  },
  async createBooking(payload) {
    return apiRequest("/api/v1/bookings", {
      method: "POST",
      body: payload,
    });
  },
  async cancelBooking(bookingId) {
    return apiRequest(`/api/v1/bookings/${bookingId}`, {
      method: "DELETE",
    });
  },
};
