"use client";

import Link from "next/link";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  ReferenceLine,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import { useAppData } from "@/context/AppDataContext";

// Meta-Informationen definieren Farbakzente für Diagramme je Raumtyp.
const CATEGORY_META = {
  "Lesesäle": {
    gradient: "from-sky-400 via-sky-500 to-sky-600",
    pieFill: "#0ea5e9",
  },
  Gruppenräume: {
    gradient: "from-amber-400 via-orange-400 to-orange-500",
    pieFill: "#f97316",
  },
  Seitenbänke: {
    gradient: "from-emerald-400 via-teal-400 to-teal-500",
    pieFill: "#10b981",
  },
};

// Vereinfacht simulierte Tagesverläufe für Chartdarstellungen.
const baseTimeline = ["08:00", "10:00", "12:00", "14:00", "16:00", "18:00", "20:00", "22:00"];
const timelineFactors = [0.25, 0.42, 0.58, 0.74, 0.85, 0.7, 0.55, 0.38];

export default function RoomDetailClient({ room }) {
  const { favorites, toggleFavorite } = useAppData();
  const isFavorite = favorites.includes(room?.id ?? -1);

  if (!room) {
    return null;
  }

  const meta = CATEGORY_META[room.type] ?? {
    gradient: "from-slate-400 via-slate-500 to-slate-600",
    pieFill: "#38bdf8",
  };

  const availability = Math.max(room.capacity - room.people, 0);
  const utilization = Math.round((room.people / room.capacity) * 100);
  const pieData = [
    { name: "Belegt", value: room.people },
    { name: "Frei", value: availability },
  ];

  const now = new Date();
  const currentHour = now.getHours();

  // Leitet aus der Zeitachse Live-, Prognose- und Erwartungswerte für das Balkendiagramm ab.
  const chartData = baseTimeline.map((time, index) => {
    const slotHour = Number(time.split(":")[0]);
    const nextHour = index === baseTimeline.length - 1 ? 24 : Number(baseTimeline[index + 1].split(":")[0]);
    let phase = "future";
    if (currentHour >= nextHour) {
      phase = "past";
    } else if (currentHour >= slotHour) {
      phase = "current";
    }
    const expected = Math.min(room.capacity, Math.round(room.capacity * timelineFactors[index]));
    const live =
      phase === "current"
        ? room.people
        : phase === "past"
        ? Math.min(expected, Math.round(expected * 0.9))
        : 0;
    const forecast = Math.max(expected - live, 0);
    return {
      time,
      live,
      forecast,
      expected,
      phase,
      isCurrent: phase === "current",
    };
  });

  // Beispielhafte zukünftige Zeitslots zur Visualisierung von Verfügbarkeiten.
  const upcomingSlots = [
    { offsetHours: 1, durationMinutes: 60, load: 0.45 },
    { offsetHours: 2, durationMinutes: 60, load: 0.6 },
    { offsetHours: 3, durationMinutes: 90, load: 0.4 },
  ].map((slot, index) => {
    const startTime = new Date(now.getTime() + slot.offsetHours * 60 * 60 * 1000);
    const endTime = new Date(startTime.getTime() + slot.durationMinutes * 60 * 1000);
    const expected = Math.min(room.capacity, Math.round(room.capacity * slot.load));
    const free = Math.max(room.capacity - expected, 0);
    return {
      id: index,
      start: startTime.toLocaleTimeString("de-DE", { hour: "2-digit", minute: "2-digit" }),
      end: endTime.toLocaleTimeString("de-DE", { hour: "2-digit", minute: "2-digit" }),
      expected,
      free,
    };
  });

  const today = now.toLocaleDateString("de-DE", {
    weekday: "long",
    day: "numeric",
    month: "long",
  });

  const statusBadge = (() => {
    // Einstufung der Auslastung für das Badging im Header.
    const ratio = room.people / room.capacity;
    if (ratio === 0) return { label: "Leer", color: "bg-slate-200 text-slate-800" };
    if (ratio <= 1 / 3) return { label: "Gering", color: "bg-emerald-500 text-white" };
    if (ratio <= 2 / 3) return { label: "Mittel", color: "bg-amber-400 text-slate-900" };
    if (ratio < 1) return { label: "Hoch", color: "bg-orange-500 text-white" };
    return { label: "Voll", color: "bg-rose-500 text-white" };
  })();

  // Nutzerdefiniertes Tooltip für die Recharts-Komponenten.
  const renderTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    const entry = payload[0].payload;
    const total = entry.live + entry.forecast;
    return (
      <div className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-600 shadow-xl">
        <p className="text-xs uppercase tracking-wide text-slate-400">{label} Uhr</p>
        <p className="mt-1 font-semibold text-slate-900">{total} Personen erwartet</p>
        {entry.live > 0 && <p className="text-xs text-slate-500">Live: {entry.live} Personen</p>}
        {entry.forecast > 0 && entry.phase !== "past" && (
          <p className="text-xs text-slate-500">Prognose: {total} Personen</p>
        )}
      </div>
    );
  };

  return (
    <div className="mx-auto flex w-full max-w-4xl flex-col gap-8 px-4 pb-8 pt-6 sm:px-6 sm:pt-8">
      <div className="flex items-center gap-3 text-sm text-sky-600">
        <Link
          href="/"
          className="inline-flex items-center gap-2 rounded-full border border-sky-200 bg-white px-4 py-2 font-medium shadow-sm transition hover:border-sky-300 hover:text-sky-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-400"
        >
          <span aria-hidden="true">←</span>
          Zur Übersicht
        </Link>
      </div>

      <section className="rounded-3xl bg-gradient-to-br from-white via-slate-50 to-sky-50 p-6 shadow-sm ring-1 ring-sky-100 sm:p-8">
        <div className="flex flex-col gap-6 sm:flex-row sm:items-start sm:justify-between">
          <div className="space-y-4">
            <span className={`inline-flex w-fit items-center gap-2 rounded-full bg-gradient-to-r ${meta.gradient} px-4 py-1 text-xs font-semibold uppercase tracking-wide text-white shadow`}>
              {room.type}
            </span>
            <div>
              <h1 className="text-2xl font-semibold text-slate-900 sm:text-3xl">{room.name}</h1>
              <p className="mt-2 text-sm text-slate-600">
                Live-Auslastung und Prognose für den heutigen Tag – aktualisiert {today}.
              </p>
            </div>
            <div className="flex flex-wrap gap-3 text-xs text-slate-500">
              <span className={`rounded-full px-3 py-1 font-semibold ${statusBadge.color}`}>{statusBadge.label}</span>
              <span className="rounded-full bg-white px-3 py-1 font-semibold text-slate-700">{utilization}% ausgelastet</span>
              <span className="rounded-full bg-white px-3 py-1 font-semibold text-slate-700">{availability} freie Plätze</span>
            </div>
            <div className="flex flex-wrap gap-3 text-sm">
              <button className="inline-flex items-center gap-2 rounded-full bg-sky-500 px-4 py-2 font-semibold text-white shadow-sm transition hover:bg-sky-600">
                Jetzt buchen
              </button>
              <button
                type="button"
                onClick={() => toggleFavorite(room.id)}
                aria-pressed={isFavorite}
                className={`inline-flex items-center gap-2 rounded-full border px-4 py-2 font-semibold shadow-sm transition ${
                  isFavorite
                    ? "border-rose-200 bg-rose-50 text-rose-600 hover:border-rose-300 hover:text-rose-700"
                    : "border-slate-200 bg-white text-slate-700 hover:border-sky-200 hover:text-sky-700"
                }`}
              >
                {isFavorite ? "Gemerkter Raum" : "Merken"}
              </button>
            </div>
          </div>
          <div className="flex flex-col items-center gap-4 sm:flex-row">
            <div className="relative h-32 w-32">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={pieData} dataKey="value" innerRadius={48} outerRadius={64} startAngle={90} endAngle={450}>
                    {pieData.map((entry, index) => (
                      <Cell key={`detail-pie-${entry.name}`} fill={index === 0 ? meta.pieFill : "#e2e8f0"} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
              <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                <span className="text-xs text-slate-500">Belegung</span>
                <span className="text-xl font-semibold text-slate-900">{utilization}%</span>
              </div>
            </div>
            <div className="grid gap-3 text-sm text-slate-600">
              <div className="rounded-2xl bg-white/90 px-4 py-3 shadow-inner">
                <p className="text-xs uppercase tracking-wide text-slate-400">Personen im Raum</p>
                <p className="mt-1 text-lg font-semibold text-slate-900">{room.people}</p>
              </div>
              <div className="rounded-2xl bg-white/90 px-4 py-3 shadow-inner">
                <p className="text-xs uppercase tracking-wide text-slate-400">Kapazität</p>
                <p className="mt-1 text-lg font-semibold text-slate-900">{room.capacity}</p>
              </div>
              <div className="rounded-2xl bg-white/90 px-4 py-3 shadow-inner">
                <p className="text-xs uppercase tracking-wide text-slate-400">Freie Plätze</p>
                <p className="mt-1 text-lg font-semibold text-slate-900">{availability}</p>
              </div>
            </div>
          </div>
        </div>
        <div className="mt-6 h-2 w-full rounded-full bg-slate-200">
          <div
            className={`h-full rounded-full bg-gradient-to-r ${meta.gradient}`}
            style={{ width: `${Math.min(utilization, 100)}%` }}
            aria-hidden="true"
          />
        </div>
      </section>

      <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-baseline sm:justify-between">
          <div>
            <h2 className="text-lg font-semibold text-slate-900 sm:text-xl">Tagesverlauf & Prognose</h2>
            <p className="text-sm text-slate-600">Vergleich von Live-Daten und erwarteter Auslastung.</p>
          </div>
          <span className="text-xs uppercase tracking-wide text-slate-400">Stand: {today}</span>
        </div>
        <div className="mt-6 h-72 w-full sm:h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} barCategoryGap={24}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="time" stroke="#64748b" />
              <YAxis stroke="#64748b" allowDecimals={false} domain={[0, room.capacity]} />
              <Tooltip content={renderTooltip} cursor={{ fill: "#f8fafc" }} />
              <ReferenceLine
                y={room.capacity}
                stroke="#94a3b8"
                strokeDasharray="4 4"
                label={{ position: "top", value: "Kapazität", fill: "#64748b", fontSize: 12 }}
              />
              <Bar dataKey="forecast" stackId="util" name="Prognose" radius={[6, 6, 0, 0]}>
                {chartData.map((entry) => (
                  <Cell
                    key={`forecast-${entry.time}`}
                    fill={entry.isCurrent ? "#bae6fd" : entry.phase === "future" ? "#e0f2fe" : "#e2e8f0"}
                  />
                ))}
              </Bar>
              <Bar dataKey="live" stackId="util" name="Live" radius={[6, 6, 0, 0]}>
                {chartData.map((entry) => (
                  <Cell
                    key={`live-${entry.time}`}
                    fill={meta.pieFill}
                    fillOpacity={entry.live > 0 ? 1 : 0}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-baseline sm:justify-between">
          <div>
            <h2 className="text-lg font-semibold text-slate-900 sm:text-xl">Freie Slots buchen</h2>
            <p className="text-sm text-slate-600">Wähle ein passendes Zeitfenster, um diesen Raum zu reservieren.</p>
          </div>
          <button className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2 text-xs font-semibold text-slate-600 shadow-sm transition hover:border-sky-200 hover:text-sky-700">
            Alle Buchungen ansehen
          </button>
        </div>
        <div className="mt-5 grid gap-3 sm:grid-cols-3">
          {upcomingSlots.map((slot) => (
            <button
              key={slot.id}
              type="button"
              className="flex flex-col gap-3 rounded-2xl border border-slate-200 bg-slate-50/80 px-4 py-4 text-left text-sm text-slate-600 shadow-sm transition hover:border-sky-200 hover:bg-white"
            >
              <div>
                <p className="text-sm font-semibold text-slate-900">
                  {slot.start} – {slot.end} Uhr
                </p>
                <p className="text-xs uppercase tracking-wide text-slate-500">{slot.expected} Personen erwartet</p>
              </div>
              <div className="flex items-center justify-between text-xs font-semibold">
                <span className="text-emerald-600">{slot.free} freie Plätze</span>
                <span className="rounded-full bg-slate-900/80 px-3 py-1 text-white">Slot wählen</span>
              </div>
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}
