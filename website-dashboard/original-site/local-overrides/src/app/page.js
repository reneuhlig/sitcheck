"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line, XAxis, YAxis, Tooltip } from "recharts";
import { rooms } from "@/data/rooms";
import { useAppData } from "@/context/AppDataContext";

// Farbschemata für die drei Raumcluster.
const CATEGORY_META = {
  "Lesesäle": {
    gradient: "from-sky-400 via-sky-500 to-sky-600",
    accent: "text-sky-600",
    pieFill: "#0ea5e9",
  },
  Gruppenräume: {
    gradient: "from-amber-400 via-orange-400 to-orange-500",
    accent: "text-orange-500",
    pieFill: "#f97316",
  },
  Seitenbänke: {
    gradient: "from-emerald-400 via-teal-400 to-teal-500",
    accent: "text-emerald-500",
    pieFill: "#10b981",
  },
};

const CATEGORY_RANGE_MULTIPLIERS = {
  "Lesesäle": {
    today: 0.68,
    tomorrow: 0.54,
  },
  Gruppenräume: {
    today: 0.58,
    tomorrow: 0.48,
  },
  Seitenbänke: {
    today: 0.5,
    tomorrow: 0.36,
  },
};

const RANGE_OPTIONS = ["live", "today", "tomorrow"];
const RANGE_LABELS = {
  live: "Live",
  today: "Heute",
  tomorrow: "Morgen",
};

const parseTimeToMinutes = (time) => {
  const [hours, minutes] = time.split(":").map(Number);
  return hours * 60 + minutes;
};

// Sichtbares Zeitformat für die Mini-Cards.
const formatTimeLabel = (isoString) =>
  new Date(isoString).toLocaleTimeString("de-DE", { hour: "2-digit", minute: "2-digit" });

const formatDateTimeLabel = (isoString) => {
  if (!isoString) return "–";
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) return "–";
  return date.toLocaleString("de-DE", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
};

export default function HomePage() {
  const { bookings } = useAppData();
  const [selectedRanges, setSelectedRanges] = useState({});
  const [hubOverview, setHubOverview] = useState(null);
  const [hubError, setHubError] = useState("");

  const groupedRooms = Object.entries(
    rooms.reduce((acc, room) => {
      if (!acc[room.type]) acc[room.type] = [];
      acc[room.type].push(room);
      return acc;
    }, {}),
  );

  const [currentPersons, setCurrentPersons] = useState(null);
  const [occupancyHistory, setOccupancyHistory] = useState([]);
  const normalizedCurrentPersons = useMemo(
    () =>
      typeof currentPersons === "number" && Number.isFinite(currentPersons)
        ? currentPersons
        : null,
    [currentPersons],
  );
  const totalCapacity = useMemo(
    () => rooms.reduce((sum, room) => sum + room.capacity, 0),
    [],
  );
  const freeSeats = useMemo(
    () => {
      if (normalizedCurrentPersons === null) {
        return null;
      }
      return Math.max(totalCapacity - normalizedCurrentPersons, 0);
    },
    [totalCapacity, normalizedCurrentPersons],
  );
  const averageUtilization = useMemo(
    () => {
      if (!totalCapacity || normalizedCurrentPersons === null) {
        return null;
      }
      return Math.round((normalizedCurrentPersons / totalCapacity) * 100);
    },
    [totalCapacity, normalizedCurrentPersons],
  );
  const serviceStatusItems = useMemo(() => {
    const services = hubOverview?.services ?? {};
    return [
      { key: "portal", label: "Portal", status: services.portal?.status ?? "unknown" },
      { key: "realtime", label: "Realtime", status: services.realtime?.status ?? "unknown" },
      { key: "api_gateway", label: "API", status: services.api_gateway?.status ?? "unknown" },
      { key: "analytics", label: "Analytics", status: services.analytics?.status ?? "unknown" },
      { key: "command_center", label: "Command Center", status: services.command_center?.status ?? "unknown" },
    ];
  }, [hubOverview]);

  useEffect(() => {
    // Nutzt primär den neuen Hub-Endpunkt und fällt bei Bedarf auf Legacy-Occupancy zurück.
    const abortController = new AbortController();
    const envBaseUrl = process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, "");
    let baseUrl = envBaseUrl;

    if (!baseUrl) {
      const defaultDevBaseUrl = "http://localhost:5000";

      if (typeof window !== "undefined") {
        const browserBaseUrl = window.location.origin.replace(/\/$/, "");
        baseUrl =
          process.env.NODE_ENV === "development"
            ? defaultDevBaseUrl
            : browserBaseUrl;
      } else {
        baseUrl = process.env.NODE_ENV === "development" ? defaultDevBaseUrl : "";
      }
    }

    baseUrl = baseUrl || "http://localhost:5000";

    async function fetchLegacyOccupancy() {
      try {
        const response = await fetch(`${baseUrl}/api/occupancy`, {
          signal: abortController.signal,
        });

        if (!response.ok) {
          throw new Error(`Request failed with status ${response.status}`);
        }

        const data = await response.json();
        if (typeof data.currentPersons === "number") {
          setCurrentPersons(data.currentPersons);
        } else if (typeof data.averagePersons === "number") {
          setCurrentPersons(Math.round(data.averagePersons));
        }
        if (Array.isArray(data.history)) {
          setOccupancyHistory(data.history);
        } else {
          setOccupancyHistory([]);
        }
        setHubError("Hub-Endpunkt nicht verfügbar, Legacy-Daten aktiv.");
      } catch (error) {
        if (error.name !== "AbortError") {
          setHubError("Hub und Legacy-Occupancy sind aktuell nicht erreichbar.");
        }
      }
    }

    async function fetchHubOverview() {
      try {
        const response = await fetch(`${baseUrl}/api/hub/overview`, {
          signal: abortController.signal,
        });

        if (!response.ok) {
          throw new Error(`Request failed with status ${response.status}`);
        }

        const data = await response.json();
        setHubOverview(data);
        setHubError("");

        const occupancyData = data?.occupancy ?? {};
        if (typeof occupancyData.currentPersons === "number") {
          setCurrentPersons(occupancyData.currentPersons);
        } else if (typeof occupancyData.averagePersons === "number") {
          setCurrentPersons(Math.round(occupancyData.averagePersons));
        }
        if (Array.isArray(occupancyData.history)) {
          setOccupancyHistory(occupancyData.history);
        } else {
          setOccupancyHistory([]);
        }
      } catch (error) {
        if (error.name === "AbortError") {
          return;
        }
        await fetchLegacyOccupancy();
      }
    }

    fetchHubOverview();
    const intervalId = setInterval(fetchHubOverview, 5 * 1000);

    return () => {
      abortController.abort();
      clearInterval(intervalId);
    };
  }, []);

  // Bereite die Buchungslisten inklusive Anzeigeformaten vor.
  const normalizedBookings = useMemo(() => {
    const now = new Date();
    return bookings
      .filter((booking) => new Date(booking.endDateTime) >= now)
      .map((booking) => ({
        ...booking,
        start: formatTimeLabel(booking.startDateTime),
        end: formatTimeLabel(booking.endDateTime),
      }));
  }, [bookings]);

  const sortedBookings = useMemo(
    () =>
      [...normalizedBookings].sort(
        (a, b) => parseTimeToMinutes(a.start) - parseTimeToMinutes(b.start),
      ),
    [normalizedBookings],
  );
  const nextBooking = sortedBookings[0];
  const additionalBookings = sortedBookings.slice(1);
  const hasUpcomingBooking = Boolean(nextBooking);
  const nextBookingDateLabel = useMemo(() => {
    if (!nextBooking) return "";
    return new Date(nextBooking.startDateTime).toLocaleDateString("de-DE", {
      weekday: "long",
      day: "numeric",
      month: "long",
    });
  }, [nextBooking]);

  const categoryTotals = groupedRooms.reduce((acc, [type, typeRooms]) => {
    const totals = typeRooms.reduce(
      (tot, room) => {
        tot.people += room.people;
        tot.capacity += room.capacity;
        return tot;
      },
      { people: 0, capacity: 0 },
    );
    acc[type] = totals;
    return acc;
  }, {});

  // Bereite die Historie sowohl für Liste als auch Diagramm auf.
  const historyChronological = useMemo(() => {
    if (!occupancyHistory.length) return [];
    return occupancyHistory
      .slice(-10)
      .map((entry) => {
        const date = new Date(entry.timestamp);
        return {
          ...entry,
          timeLabel: date.toLocaleTimeString("de-DE", {
            hour: "2-digit",
            minute: "2-digit",
          }),
          dateLabel: date.toLocaleDateString("de-DE", {
            weekday: "short",
            day: "2-digit",
            month: "2-digit",
          }),
        };
      });
  }, [occupancyHistory]);

  const recentHistory = useMemo(
    () => [...historyChronological].reverse(),
    [historyChronological],
  );

  // Recharts benötigt ein kompaktes Array mit X-Achsen-Labeln.
  const historyChartData = useMemo(
    () =>
      historyChronological.map((entry) => ({
        name: entry.timeLabel,
        persons: entry.persons,
      })),
    [historyChronological],
  );

  return (
    <div className="mx-auto flex w-full max-w-5xl flex-col gap-10 pb-6 sm:gap-12">
      <section className="space-y-4">
        <div className="rounded-4xl bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-6 text-white shadow-lg sm:p-8">
          <div className="flex flex-col gap-5 sm:flex-row sm:items-start sm:justify-between">
            <div className="space-y-3">
              <span className="inline-flex items-center gap-2 rounded-full bg-white/10 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-sky-200">
                Portal-first Hauptzugang
              </span>
              <h1 className="text-2xl font-semibold sm:text-3xl">Sitcheck Operations Hub</h1>
              <p className="max-w-2xl text-sm text-slate-200">
                Zentrale Übersicht für Realtime, API und Analytics auf einem Einstiegspunkt.
              </p>
            </div>
            <div className="grid gap-2 text-sm">
              <a
                href="/realtime"
                className="rounded-xl border border-white/20 bg-white/10 px-4 py-2 font-semibold text-white transition hover:bg-white/20"
              >
                Realtime öffnen
              </a>
              <a
                href="/analytics"
                className="rounded-xl border border-white/20 bg-white/10 px-4 py-2 font-semibold text-white transition hover:bg-white/20"
              >
                Analytics öffnen
              </a>
              <a
                href="/api/v1/dashboard/command-center?zone_id=default-zone&horizon=60&history_minutes=180"
                className="rounded-xl border border-white/20 bg-white/10 px-4 py-2 font-semibold text-white transition hover:bg-white/20"
              >
                Command Center
              </a>
            </div>
          </div>

          {hubError && (
            <div className="mt-4 rounded-2xl border border-amber-300/40 bg-amber-200/20 px-4 py-3 text-sm text-amber-100">
              {hubError}
            </div>
          )}

          <div className="mt-6 grid grid-cols-1 gap-3 text-sm sm:grid-cols-2 lg:grid-cols-5">
            {[
              { label: "Hub-Status", value: hubOverview?.status ?? "unknown" },
              { label: "Personen live", value: normalizedCurrentPersons ?? "–" },
              { label: "Durchschnitt", value: hubOverview?.occupancy?.averagePersons ?? "–" },
              { label: "Realtime FPS", value: hubOverview?.realtime_state?.fps ?? "–" },
              { label: "Tracks", value: hubOverview?.realtime_state?.tracks ?? "–" },
            ].map((item) => (
              <div key={item.label} className="rounded-2xl border border-white/15 bg-white/10 px-4 py-3">
                <p className="text-[11px] uppercase tracking-wide text-slate-300">{item.label}</p>
                <p className="mt-2 text-lg font-semibold text-white">{item.value}</p>
              </div>
            ))}
          </div>

          <div className="mt-4 grid grid-cols-2 gap-2 text-xs sm:grid-cols-3 lg:grid-cols-5">
            {serviceStatusItems.map((service) => {
              const status = String(service.status || "unknown").toLowerCase();
              const badgeClass =
                status === "ok"
                  ? "border-emerald-300/40 bg-emerald-300/20 text-emerald-100"
                  : status === "degraded"
                    ? "border-amber-300/40 bg-amber-300/20 text-amber-100"
                    : "border-rose-300/40 bg-rose-300/20 text-rose-100";
              return (
                <div key={service.key} className={`rounded-xl border px-3 py-2 ${badgeClass}`}>
                  <p className="font-semibold">{service.label}</p>
                  <p className="uppercase tracking-wide">{service.status}</p>
                </div>
              );
            })}
          </div>
          <p className="mt-3 text-xs text-slate-300">
            Letztes Hub-Update: {formatDateTimeLabel(hubOverview?.generated_at ?? hubOverview?.occupancy?.lastUpdated)}
          </p>
        </div>
      </section>

      <section className="space-y-4 sm:space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
          <h2 className="text-lg font-semibold text-slate-900">Campus-Demo Bereich</h2>
          <span className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
            Buchungen und Favoriten bleiben lokal/in-memory
          </span>
        </div>
        <div className="rounded-4xl bg-gradient-to-br from-sky-100 via-sky-50 to-white p-6 shadow-sm ring-1 ring-sky-100 sm:p-8">
          {hasUpcomingBooking ? (
            <>
              <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
                <div className="space-y-3">
                  <span className="inline-flex items-center gap-2 rounded-full bg-white/70 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-sky-700">
                    Meine nächste Buchung
                  </span>
                  <div>
                    <h1 className="text-2xl font-semibold text-slate-900 sm:text-3xl">{nextBooking.roomName}</h1>
                    <p className="mt-1 text-sm text-slate-600">
                      {nextBooking.start} – {nextBooking.end} Uhr · {nextBooking.category}
                    </p>
                  </div>
                  <div className="flex flex-wrap gap-3 text-xs text-slate-500">
                    <span className="rounded-full bg-sky-200/50 px-3 py-1 font-medium text-sky-700">Status: {nextBooking.status}</span>
                    <span className="rounded-full bg-white px-3 py-1 font-medium text-slate-700">{nextBookingDateLabel}</span>
                  </div>
                </div>
                <div className="flex flex-col gap-3 text-sm text-slate-600">
                  <Link
                    href="/bookings"
                    className="inline-flex items-center justify-center gap-2 rounded-full bg-sky-500 px-4 py-2 font-semibold text-white shadow-sm transition hover:bg-sky-600"
                  >
                    Buchung verwalten
                    <span aria-hidden="true">→</span>
                  </Link>
                  <Link
                    href="/bookings/new"
                    className="inline-flex items-center justify-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2 font-semibold text-slate-700 shadow-sm transition hover:border-sky-200 hover:text-sky-700"
                  >
                    Neuen Raum suchen
                  </Link>
                </div>
              </div>

              {additionalBookings.length > 0 && (
                <div className="mt-6 grid gap-3 sm:grid-cols-2">
                  {additionalBookings.map((booking) => (
                    <div
                      key={booking.id}
                      className="flex items-center justify-between rounded-3xl border border-slate-200 bg-white/70 px-4 py-3 text-sm text-slate-600 shadow-inner"
                    >
                      <div>
                        <p className="text-sm font-semibold text-slate-900">{booking.roomName}</p>
                        <p className="text-xs uppercase tracking-wide text-slate-500">
                          {booking.start} – {booking.end} Uhr · {booking.category}
                        </p>
                      </div>
                      <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-600">
                        {booking.status}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </>
          ) : (
            <div className="flex flex-col items-center gap-4 text-center text-slate-600">
              <span className="inline-flex items-center gap-2 rounded-full bg-white/70 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-sky-700">
                Keine offenen Buchungen
              </span>
              <h2 className="text-xl font-semibold text-slate-900 sm:text-2xl">Starte deine erste Reservierung</h2>
              <p className="max-w-md text-sm">
                Aktuell sind keine Buchungen geplant. Du kannst jederzeit einen Raum auswählen und eine Anfrage anlegen.
              </p>
              <Link
                href="/bookings/new"
                className="inline-flex items-center gap-2 rounded-full bg-sky-500 px-4 py-2 font-semibold text-white shadow-sm transition hover:bg-sky-600"
              >
                Jetzt unverbindlich buchen
                <span aria-hidden="true">→</span>
              </Link>
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 gap-3 text-sm text-slate-600 sm:grid-cols-3">
          {[
            { label: "Aktive Räume", value: rooms.length },
            { label: "Personen aktuell", value: normalizedCurrentPersons },
            { label: "Freie Plätze", value: freeSeats },
            { label: "Ø Auslastung", value: averageUtilization, suffix: "%" },
          ].map((stat) => {
            const displayValue =
              stat.value === null || typeof stat.value === "undefined"
                ? "–"
                : `${stat.value}${stat.suffix ?? ""}`;
            return (
              <div
                key={stat.label}
                className="rounded-3xl border border-slate-200 bg-white px-5 py-4 shadow-sm"
              >
                <p className="text-xs uppercase tracking-wide text-slate-400">{stat.label}</p>
                <p className="mt-3 text-2xl font-semibold text-slate-900">{displayValue}</p>
              </div>
            );
          })}
        </div>

        <div className="rounded-3xl border border-slate-200 bg-white px-5 py-4 shadow-sm">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <p className="text-xs uppercase tracking-wide text-slate-400">Historie · Personen aktuell</p>
              <p className="text-sm text-slate-500">Letzte Messpunkte aus dem Live-Zähler.</p>
            </div>
            <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">
              {recentHistory.length} Einträge
            </span>
          </div>
          {recentHistory.length > 0 ? (
            <>
              {/* Kompakter Linienchart als visueller Verlauf */}
              <div className="mb-6 w-full overflow-hidden rounded-2xl border border-slate-100 bg-slate-50/60 p-3">
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={historyChartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                    <XAxis dataKey="name" tick={{ fontSize: 12 }} stroke="#94a3b8" />
                    <YAxis allowDecimals={false} width={30} stroke="#94a3b8" />
                    <Tooltip
                      labelFormatter={(label) => `Zeit: ${label}`}
                      formatter={(value) => [`${value} Personen`, "Live-Wert"]}
                    />
                    <Line type="monotone" dataKey="persons" stroke="#0ea5e9" strokeWidth={3} dot={{ r: 4 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="grid gap-2 text-sm text-slate-600 sm:grid-cols-2">
                {recentHistory.map((entry) => (
                  <div
                    key={`${entry.timestamp}-${entry.persons}`}
                    className="flex items-center justify-between rounded-2xl border border-slate-100 bg-slate-50/60 px-4 py-2"
                  >
                    <div className="flex flex-col">
                      <span className="text-xs uppercase tracking-wide text-slate-400">{entry.timeLabel} Uhr</span>
                      <span className="text-sm font-semibold text-slate-900">{entry.dateLabel}</span>
                    </div>
                    <span className="text-xl font-semibold text-slate-900">{entry.persons}</span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <p className="text-sm text-slate-500">Noch keine historischen Messwerte erhalten.</p>
          )}
        </div>
      </section>

      <section className="space-y-8">
        {groupedRooms.map(([type, typeRooms]) => {
          const totals = categoryTotals[type];
          const meta = CATEGORY_META[type] ?? {
            gradient: "from-slate-400 via-slate-500 to-slate-600",
            accent: "text-slate-600",
            pieFill: "#64748b",
          };
          const activeRange = selectedRanges[type] ?? "live";
          const multiplier = CATEGORY_RANGE_MULTIPLIERS[type]?.[activeRange] ?? 0.6;
          const projectedOccupied =
            activeRange === "live"
              ? totals.people
              : Math.round(totals.capacity * multiplier);
          const occupiedSeats = Math.min(projectedOccupied, totals.capacity);
          const freeSeatsRange = Math.max(totals.capacity - occupiedSeats, 0);
          const utilization = totals.capacity
            ? Math.round((occupiedSeats / totals.capacity) * 100)
            : 0;
          const pieData = [
            { name: "Belegt", value: occupiedSeats },
            { name: "Frei", value: freeSeatsRange },
          ];

          return (
            <article
              key={type}
              className="rounded-4xl border border-slate-200 bg-white/90 p-6 shadow-sm backdrop-blur sm:p-8"
            >
              <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
                <div className="space-y-3">
                  <div className="flex items-center gap-3">
                    <span
                      className={`inline-flex items-center gap-2 rounded-full bg-gradient-to-r ${meta.gradient} px-4 py-1 text-xs font-semibold uppercase tracking-wide text-white shadow`}
                    >
                      {type}
                    </span>
                    <span className="text-xs font-medium uppercase tracking-wide text-slate-500">
                      {typeRooms.length} Räume
                    </span>
                  </div>
                  <p className="text-sm text-slate-600">
                    Aggregierte Auslastung aller {type} im Überblick.
                  </p>
                    <div className="flex flex-wrap gap-4 text-xs text-slate-500">
                      <span>
                        Belegt: <span className="font-semibold text-slate-900">{occupiedSeats}</span>
                      </span>
                      <span>
                        Kapazität: <span className="font-semibold text-slate-900">{totals.capacity}</span>
                      </span>
                      <span>
                        Frei: <span className="font-semibold text-slate-900">{freeSeatsRange}</span>
                      </span>
                    </div>
                  </div>
                  <div className="flex flex-col items-center gap-4 sm:flex-row">
                    <div className="flex items-center gap-2 rounded-full border border-slate-200 bg-white px-2 py-1 text-xs font-semibold text-slate-600">
                      {RANGE_OPTIONS.map((range) => {
                        const isActive = activeRange === range;
                        return (
                          <button
                            key={range}
                            type="button"
                            onClick={() =>
                              setSelectedRanges((prev) => ({
                                ...prev,
                                [type]: range,
                              }))
                            }
                            className={`rounded-full px-2.5 py-1 transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-500 ${
                              isActive
                                ? "bg-sky-500 text-white shadow-sm"
                                : "text-slate-500 hover:text-slate-700"
                            }`}
                            aria-pressed={isActive}
                          >
                            {RANGE_LABELS[range]}
                          </button>
                        );
                      })}
                    </div>
                    <div className="relative h-28 w-28">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                        <Pie
                          data={pieData}
                          dataKey="value"
                          innerRadius={40}
                          outerRadius={56}
                          startAngle={90}
                          endAngle={450}
                        >
                          {pieData.map((entry, index) => (
                            <Cell
                              key={`${type}-pie-${entry.name}`}
                              fill={index === 0 ? meta.pieFill : "#e2e8f0"}
                            />
                          ))}
                        </Pie>
                      </PieChart>
                    </ResponsiveContainer>
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                      <span className="text-xs text-slate-500">Auslastung</span>
                      <span className="text-lg font-semibold text-slate-900">{utilization}%</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="-mx-5 mt-6 overflow-x-auto pb-4">
                <div className="flex gap-4 px-1 sm:px-5">
                  {typeRooms.map((room) => {
                    const statusRatio = room.people / room.capacity;
                    const status = (() => {
                      if (statusRatio === 0)
                        return { label: "Leer", badge: "bg-slate-200 text-slate-800" };
                      if (statusRatio <= 1 / 3)
                        return { label: "Gering", badge: "bg-emerald-500 text-white" };
                      if (statusRatio <= 2 / 3)
                        return { label: "Mittel", badge: "bg-amber-400 text-slate-900" };
                      if (statusRatio < 1)
                        return { label: "Hoch", badge: "bg-orange-500 text-white" };
                      return { label: "Voll", badge: "bg-rose-500 text-white" };
                    })();
                    const utilizationRoom = Math.round((room.people / room.capacity) * 100);

                    return (
                      <Link
                        href={`/rooms/${room.id}`}
                        key={room.id}
                        className="group flex min-w-[240px] snap-start flex-col gap-4 rounded-3xl border border-slate-200 bg-gradient-to-br from-white via-white to-slate-50 p-5 shadow-sm transition hover:-translate-y-1 hover:shadow-lg focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-400"
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div className="space-y-1">
                            <p className="text-xs uppercase tracking-wide text-slate-400">{room.type}</p>
                            <h3 className="text-lg font-semibold text-slate-900">{room.name}</h3>
                          </div>
                          <span className={`rounded-full px-3 py-1 text-xs font-semibold ${status.badge}`}>
                            {status.label}
                          </span>
                        </div>
                        <div className="grid grid-cols-2 gap-3 text-xs text-slate-500">
                          <div>
                            <p>Personen</p>
                            <p className="text-sm font-semibold text-slate-900">{room.people}</p>
                          </div>
                          <div>
                            <p>Kapazität</p>
                            <p className="text-sm font-semibold text-slate-900">{room.capacity}</p>
                          </div>
                          <div>
                            <p>Auslastung</p>
                            <p className="text-sm font-semibold text-slate-900">{utilizationRoom}%</p>
                          </div>
                          <div>
                            <p>Freie Plätze</p>
                            <p className="text-sm font-semibold text-slate-900">{Math.max(room.capacity - room.people, 0)}</p>
                          </div>
                        </div>
                        <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200">
                          <div
                            className={`h-full rounded-full bg-gradient-to-r ${meta.gradient}`}
                            style={{ width: `${Math.min(utilizationRoom, 100)}%` }}
                            aria-hidden="true"
                          />
                        </div>
                        <div className="flex items-center justify-between text-xs font-semibold text-sky-600">
                          <span className="flex items-center gap-2">
                            Details ansehen
                            <span aria-hidden="true" className="transition-transform group-hover:translate-x-1">
                              →
                            </span>
                          </span>
                          <span className="text-slate-500">{RANGE_LABELS[activeRange]}</span>
                        </div>
                      </Link>
                    );
                  })}
                </div>
              </div>
            </article>
          );
        })}
      </section>

    </div>
  );
}
