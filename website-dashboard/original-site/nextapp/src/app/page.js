"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { sitcheckApi } from "@/lib/api";
import { useAppData } from "@/context/AppDataContext";

const NARRATIVE_TIMEOUT_MS = 120000;
const FORECAST_START_OFFSET_MINUTES = 30;
const FORECAST_DURATION_MINUTES = 180;
const FORECAST_END_OFFSET_MINUTES = FORECAST_START_OFFSET_MINUTES + FORECAST_DURATION_MINUTES;
const FORECAST_BUCKET_MINUTES = 10;
const FORECAST_WINDOW_LABEL = "ab +30 Minuten für 3 Stunden";

function formatWholeNumber(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "–";
  return `${Math.round(value)}`;
}

function formatDateTime(value) {
  if (!value) return "–";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "–";
  return date.toLocaleString("de-DE", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatTime(value) {
  if (!value) return "–";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "–";
  return date.toLocaleTimeString("de-DE", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function asNumber(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return null;
  }
  return value;
}

function averageForecastValue(points, key) {
  const values = points
    .map((point) => asNumber(point?.[key]))
    .filter((value) => typeof value === "number");

  if (values.length === 0) {
    return null;
  }

  return Math.round(values.reduce((sum, value) => sum + value, 0) / values.length);
}

function buildForecastChartData(forecastPoints) {
  const points = Array.isArray(forecastPoints) ? forecastPoints : [];
  const nowMs = Math.floor(Date.now() / 60000) * 60000;
  const startMs = nowMs + FORECAST_START_OFFSET_MINUTES * 60 * 1000;
  const endMs = nowMs + FORECAST_END_OFFSET_MINUTES * 60 * 1000;
  const bucketMs = FORECAST_BUCKET_MINUTES * 60 * 1000;
  const buckets = new Map();

  points.forEach((point) => {
    const timestampMs = new Date(point?.timestamp).getTime();
    if (!Number.isFinite(timestampMs) || timestampMs < startMs || timestampMs > endMs) {
      return;
    }

    const bucketIndex = Math.floor((timestampMs - startMs) / bucketMs);
    const bucketStartMs = startMs + bucketIndex * bucketMs;

    if (!buckets.has(bucketIndex)) {
      buckets.set(bucketIndex, {
        timestamp: bucketStartMs,
        points: [],
      });
    }

    buckets.get(bucketIndex).points.push(point);
  });

  return Array.from(buckets.values())
    .sort((left, right) => left.timestamp - right.timestamp)
    .map((bucket) => ({
      label: formatTime(bucket.timestamp),
      forecast: averageForecastValue(bucket.points, "yhat"),
      lower: averageForecastValue(bucket.points, "pi_low"),
      upper: averageForecastValue(bucket.points, "pi_high"),
    }));
}

function formatCountPhrase(count, singular, plural) {
  return `${count} ${count === 1 ? singular : plural}`;
}

function normalizeDriverKey(name) {
  return String(name ?? "").trim().toLowerCase();
}

function buildDriverInterpretation(driver, context) {
  const key = normalizeDriverKey(driver?.name);
  const direction = driver?.direction === "down" ? "down" : driver?.direction === "mixed" ? "mixed" : "up";
  const { bookingCount, eventCount } = context;

  switch (key) {
    case "lecture_density":
      if (direction === "up") {
        return "Ein wichtiger Grund ist das hohe Vorlesungsaufkommen auf dem Campus. Wenn viele Lehrveranstaltungen parallel laufen, steigt erfahrungsgemäß auch der Zulauf in die Bibliothek.";
      }
      return "Das aktuelle Vorlesungsaufkommen liefert kein ganz eindeutiges Signal, bleibt aber ein relevanter Einfluss für die Bibliotheksauslastung.";
    case "lecture_start_wave":
      return "Zusätzlich deuten anstehende Vorlesungsstarts darauf hin, dass sich die Auslastung rund um den Beginn der Veranstaltungen spürbar verändern kann.";
    case "trend":
      if (direction === "up") {
        return "Der bisherige Verlauf der letzten Messwerte zeigt eher nach oben. Deshalb rechnet das Modell kurzfristig mit weiter steigender Belegung.";
      }
      if (direction === "down") {
        return "Der jüngste Verlauf zeigt eher eine Entspannung. Deshalb wird die Auslastung in der nahen Zukunft etwas vorsichtiger eingeschätzt.";
      }
      return "Der aktuelle Verlauf zeigt kein klares Auf- oder Abwärtssignal. Deshalb bleibt die kurzfristige Entwicklung eher offen.";
    case "momentum":
      if (direction === "up") {
        return "Die letzten Minuten waren dynamischer als der längere Vergleich über die letzte Stunde. Das spricht für zusätzlichen kurzfristigen Zulauf.";
      }
      if (direction === "down") {
        return "Kurzfristig hat die Dynamik zuletzt eher nachgelassen. Das dämpft den erwarteten Anstieg etwas.";
      }
      return "Kurzfristig gibt es gerade keine klare Beschleunigung oder Abschwächung der Bewegung in Richtung Bibliothek.";
    case "seasonality_proxy":
      if (direction === "up") {
        return "Auch der typische Tagesverlauf spricht im Moment eher für eine höhere Auslastung als noch vor etwa einer Stunde.";
      }
      if (direction === "down") {
        return "Verglichen mit dem üblichen Verlauf zu dieser Tageszeit spricht das aktuelle Muster eher für eine etwas niedrigere Auslastung.";
      }
      return "Der typische Tagesverlauf liefert aktuell kein starkes zusätzliches Signal nach oben oder unten.";
    case "event_context":
      if (eventCount > 0) {
        return `${eventCount} laufende oder anstehende Termine im Umfeld der Bibliothek können zusätzlich für mehr Bewegung und damit für veränderte Auslastung sorgen.`;
      }
      return "Auch das Termin- und Veranstaltungsgeschehen im Umfeld der Bibliothek kann die Auslastung kurzfristig beeinflussen.";
    case "data_quality_penalty":
      return "Die Datenqualität ist momentan nicht ideal. Deshalb formuliert das Modell die Prognose vorsichtiger und die genaue Höhe sollte eher als Tendenz gelesen werden.";
    case "buchungen":
    case "bookings":
      if (bookingCount > 0) {
        return `${bookingCount} bestätigte Buchungen sind bereits bekannte Nachfrage und erhöhen die Prognose im jeweiligen Zeitfenster zusätzlich.`;
      }
      return "Bestätigte Buchungen gehen als bekannte Nachfrage direkt in die Kurzfristprognose ein.";
    case "upstream_unavailable":
      return "Die Detailerklärung aus dem XAI-Backend ist derzeit nur eingeschränkt verfügbar. Deshalb wird die Prognose vorsichtiger und stärker als Tendenz dargestellt.";
    case "shap_placeholder":
      return "Die Ursachenanalyse basiert aktuell auf vereinfachten, aber nachvollziehbaren Einflussfaktoren statt auf einer tieferen Merkmalszerlegung.";
    default:
      if (direction === "up") {
        return "Ein weiterer Modellfaktor spricht derzeit eher für steigende Auslastung.";
      }
      if (direction === "down") {
        return "Ein weiterer Modellfaktor wirkt momentan eher dämpfend auf die erwartete Auslastung.";
      }
      return "Ein weiterer Modellfaktor liefert aktuell kein eindeutiges Signal.";
  }
}

function buildUncertaintyNarrative(level, drivers) {
  const normalizedLevel = String(level ?? "").toLowerCase();
  const hasQualityPenalty = drivers.some((driver) => normalizeDriverKey(driver?.name) === "data_quality_penalty");
  const hasUpstreamIssue = drivers.some((driver) => normalizeDriverKey(driver?.name) === "upstream_unavailable");

  if (normalizedLevel === "high") {
    return hasQualityPenalty || hasUpstreamIssue
      ? "Die Unsicherheit ist aktuell hoch. Das heißt: Die Richtung der Prognose kann sinnvoll sein, aber die genaue Höhe ist wegen eingeschränkter Datenlage deutlich unsicherer."
      : "Die Unsicherheit ist aktuell hoch. Das Modell sieht also eher eine grobe Tendenz als einen sehr stabilen exakten Wert.";
  }

  if (normalizedLevel === "medium") {
    return hasQualityPenalty
      ? "Die Unsicherheit ist aktuell mittel. Die Grundrichtung der Prognose ist plausibel, aber wegen der Datenqualität kann sich die genaue Höhe noch merklich verschieben."
      : "Die Unsicherheit ist aktuell mittel. Die Richtung der Prognose ist plausibel, die genaue Höhe kann sich aber noch spürbar verändern.";
  }

  return "Die Unsicherheit ist derzeit eher niedrig. Das bedeutet nicht völlige Sicherheit, aber die Prognose wirkt im Moment vergleichsweise stabil.";
}

function getPointOccupancyValue(point) {
  const occupancy = asNumber(point?.occupancy);
  if (occupancy !== null) {
    return occupancy;
  }

  return asNumber(point?.persons);
}

function analyzeHistoryTrend(historyPoints) {
  const values = (Array.isArray(historyPoints) ? historyPoints : [])
    .map((point) => getPointOccupancyValue(point))
    .filter((value) => typeof value === "number");

  if (values.length < 2) {
    return { label: "", delta: 0 };
  }

  const windowValues = values.slice(-Math.min(values.length, 6));
  const delta = windowValues[windowValues.length - 1] - windowValues[0];

  if (delta >= 8) {
    return { label: "steigend", delta };
  }

  if (delta <= -8) {
    return { label: "fallend", delta };
  }

  return { label: "stabil", delta };
}

function buildTrendNarrative(historyPoints) {
  const trend = analyzeHistoryTrend(historyPoints);

  if (trend.label === "steigend") {
    return "Im direkten Live-Verlauf zeigt die Belegung der letzten Messpunkte klar nach oben. Das passt dazu, dass das Modell kurzfristig mit weiterem Zulauf rechnet.";
  }

  if (trend.label === "fallend") {
    return "Im direkten Live-Verlauf ist zuletzt eher Entspannung zu sehen. Deshalb wirkt die Prognose vorsichtiger als in einer klaren Wachstumssituation.";
  }

  if (trend.label === "stabil") {
    return "Im direkten Live-Verlauf schwankt die Belegung zuletzt nur moderat. Das spricht eher für eine stabile Kurzfristsituation als für einen abrupten Umschwung.";
  }

  return "";
}

function buildCapacityPressureNarrative({ currentPersons, zoneCapacity, forecastCurrent, forecastPeak }) {
  const current = asNumber(currentPersons);
  const capacity = asNumber(zoneCapacity);
  const projectedPeak = asNumber(forecastPeak) ?? asNumber(forecastCurrent);

  if (current === null || capacity === null || capacity <= 0) {
    return "";
  }

  const projectedUtilization = (projectedPeak ?? current) / capacity;

  if (projectedUtilization >= 0.9) {
    return "Aus Kapazitätssicht bewegt sich die Bibliothek gerade in Richtung eines sehr hohen Auslastungsniveaus. Schon kleine zusätzliche Nachfrageimpulse können die Platzsuche deutlich anspannen.";
  }

  if (projectedUtilization >= 0.75) {
    return "Aus Kapazitätssicht ist die Lage bereits deutlich belebt. Wenn weiterer Zulauf hinzukommt, kann sich die Atmosphäre in der nächsten Stunde spürbar verdichten.";
  }

  if (projectedUtilization <= 0.45) {
    return "Aus Kapazitätssicht wirkt die Situation aktuell noch vergleichsweise entspannt. Das Modell sieht also eher Bewegung innerhalb eines noch gut handhabbaren Bereichs.";
  }

  return "Aus Kapazitätssicht liegt die Bibliothek aktuell in einem mittleren Bereich. Die Entwicklung hängt deshalb vor allem davon ab, ob kurzfristig zusätzliche Nachfragewellen eintreffen.";
}

function buildBookingNarrative({ upcomingBookings, nextBooking }) {
  const bookingCount = Array.isArray(upcomingBookings) ? upcomingBookings.length : 0;

  if (bookingCount <= 0) {
    return "";
  }

  const nextBookingText =
    nextBooking?.starts_at && formatTime(nextBooking.starts_at) !== "–"
      ? `, die nächste startet um ${formatTime(nextBooking.starts_at)}`
      : "";

  if (bookingCount === 1) {
    return `Eine bestätigte Buchung ist bereits als bekannte Nachfrage eingerechnet${nextBookingText}.`;
  }

  return `${formatCountPhrase(bookingCount, "bestätigte Buchung", "bestätigte Buchungen")} sind bereits als bekannte Nachfrage eingerechnet${nextBookingText}.`;
}

function buildRecommendationNarrative({
  forecastPeak,
  zoneCapacity,
  freeSeats,
  uncertaintyLevel,
  upcomingBookings,
  trendLabel,
  libraryHours,
}) {
  const projectedPeak = asNumber(forecastPeak);
  const capacity = asNumber(zoneCapacity);
  const remainingSeats = asNumber(freeSeats);
  const bookingCount = Array.isArray(upcomingBookings) ? upcomingBookings.length : 0;
  const normalizedUncertainty = String(uncertaintyLevel ?? "").toLowerCase();

  if (libraryHours?.is_open_now === false) {
    return `Für Besucher bedeutet das konkret: Laut Öffnungszeiten ist die Bibliothek aktuell geschlossen${typeof libraryHours?.next_open_label === "string" ? ` und öffnet wieder ${libraryHours.next_open_label}.` : "."}`;
  }

  if (libraryHours?.closes_within_horizon && typeof libraryHours?.today_close === "string") {
    return `Für Besucher bedeutet das konkret: Die Bibliothek schließt heute um ${libraryHours.today_close} Uhr. Für einen längeren Besuch ist das in der nächsten Stunde der wichtigste praktische Faktor.`;
  }

  if (capacity !== null && projectedPeak !== null && projectedPeak >= capacity * 0.9) {
    return "Für Besucher bedeutet das konkret: Wer sicher einen Platz braucht, sollte eher früher kommen oder direkt reservieren, weil die Bibliothek in Richtung Vollauslastung läuft.";
  }

  if (remainingSeats !== null && remainingSeats <= 15) {
    return "Für Besucher bedeutet das konkret: Freie Plätze sind rechnerisch schon knapp. Ein kurzer Blick auf das nächste Update oder eine Buchung kann sich lohnen.";
  }

  if (normalizedUncertainty === "high") {
    return "Für Besucher bedeutet das konkret: Die Richtung der Prognose ist brauchbar, aber wegen der höheren Unsicherheit lohnt sich kurz vor dem Besuch noch ein frischer Blick auf das Dashboard.";
  }

  if (bookingCount > 0 && trendLabel === "steigend") {
    return "Für Besucher bedeutet das konkret: Weil Live-Trend und Buchungen beide nach oben zeigen, dürfte es in der nächsten Stunde eher schwieriger werden, spontan einen ruhigen Platz zu finden.";
  }

  return "Für Besucher bedeutet das konkret: Die 3 Stunden ab +30 Minuten wirken aktuell gut planbar, auch wenn kleinere Ausschläge im laufenden Betrieb normal bleiben.";
}

function buildLiveExplainabilitySnapshot({
  currentPersons,
  zoneCapacity,
  forecastCurrent,
  forecastPeak,
  upcomingBookings,
  calendarEvents,
  campusContext,
  historyPoints,
}) {
  const parts = [];
  const current = asNumber(currentPersons);
  const capacity = asNumber(zoneCapacity);
  const forecastValue = asNumber(forecastCurrent);
  const peakValue = asNumber(forecastPeak);
  const bookingCount = Array.isArray(upcomingBookings) ? upcomingBookings.length : 0;
  const eventCount = Array.isArray(calendarEvents) ? calendarEvents.length : 0;
  const libraryHours = campusContext?.library_hours ?? null;
  const activeCampusLectures = Math.max(0, Math.round(asNumber(campusContext?.active_onsite_lectures) ?? 0));
  const campusStartsNextHour = Math.max(0, Math.round(asNumber(campusContext?.starting_next_60m) ?? 0));
  const campusEndsNextHour = Math.max(0, Math.round(asNumber(campusContext?.ending_next_60m) ?? 0));
  const trend = analyzeHistoryTrend(historyPoints);

  if (current !== null) {
    parts.push(`${formatWholeNumber(current)} Personen sind aktuell vor Ort`);
  }

  if (current !== null && capacity !== null && capacity > 0) {
    parts.push(`das entspricht rund ${Math.round((current / capacity) * 100)}% der Kapazität`);
  }

  if (forecastValue !== null) {
    parts.push(`zum Start des Prognosebands in +30 Minuten liegt die Erwartung bei etwa ${formatWholeNumber(forecastValue)} Personen`);
  }

  if (peakValue !== null && peakValue !== forecastValue) {
    parts.push(`der Peak liegt bei ungefähr ${formatWholeNumber(peakValue)}`);
  }

  if (trend.label === "steigend") {
    parts.push("der jüngste Live-Trend zeigt nach oben");
  } else if (trend.label === "fallend") {
    parts.push("der jüngste Live-Trend entspannt sich leicht");
  }

  if (bookingCount > 0) {
    parts.push(`${formatCountPhrase(bookingCount, "bestätigte Buchung ist", "bestätigte Buchungen sind")} bereits eingepreist`);
  }

  if (libraryHours?.is_open_now === false) {
    parts.push("laut Öffnungszeiten ist die Bibliothek aktuell geschlossen");
  } else if (libraryHours?.closes_within_horizon && typeof libraryHours?.today_close === "string") {
    parts.push(`die Bibliothek schließt heute bereits um ${libraryHours.today_close} Uhr`);
  } else if (typeof libraryHours?.today_close === "string") {
    parts.push(`die Bibliothek ist heute bis ${libraryHours.today_close} Uhr geöffnet`);
  }

  if (activeCampusLectures > 0) {
    parts.push(`laut dhbw.app laufen in Mannheim aktuell ${formatWholeNumber(activeCampusLectures)} Präsenzveranstaltungen`);
  }

  if (campusStartsNextHour > 0 || campusEndsNextHour > 0) {
    parts.push(`im Mannheim-Stundenplan zeigt die nächste Stunde ${formatWholeNumber(campusStartsNextHour)} Starts und ${formatWholeNumber(campusEndsNextHour)} Enden`);
  } else if (eventCount > 0) {
    parts.push(`${eventCount} ${eventCount === 1 ? "Campus-Termin liefert" : "Campus-Termine liefern"} zusätzlichen Kontext`);
  }

  if (parts.length === 0) {
    return "";
  }

  return `Live-Kontext jetzt: ${parts.join(", ")}.`;
}

function postProcessNarrativeText(text) {
  if (typeof text !== "string" || !text.trim()) {
    return "";
  }

  return text
    .replace(/lecture_density/gi, "hohes Vorlesungsaufkommen")
    .replace(/lecture_start_wave/gi, "anstehende Vorlesungsstarts")
    .replace(/data_quality_penalty/gi, "eingeschränkte Datenqualität")
    .replace(/seasonality_proxy/gi, "typischer Tagesverlauf")
    .replace(/event_context/gi, "Termin- und Veranstaltungseinflüsse")
    .replace(/momentum/gi, "kurzfristige Dynamik")
    .replace(/\btrend\b/gi, "aktueller Verlauf")
    .replace(
      /Top demand drivers calculated with progressive disclosure: summary, drivers, evidence, and counterfactual-ready context\./gi,
      "Die Prognose wird aus dem aktuellen Verlauf, bekannten Nachfragefaktoren und erklärbaren Einflussgrößen abgeleitet.",
    )
    .replace(
      /Combines forecast interval width and data quality penalty\./gi,
      "Die Unsicherheit ergibt sich vor allem aus der Breite des Prognosebands und der aktuellen Datenqualität.",
    )
    .trim();
}

function buildLibraryHoursNarrative(libraryHours) {
  return typeof libraryHours?.statement === "string" ? libraryHours.statement.trim() : "";
}

function buildCampusContextNarrative({ campusContext, calendarEvents }) {
  const campusStatement = typeof campusContext?.statement === "string" ? campusContext.statement.trim() : "";
  const campusEvents = Array.isArray(calendarEvents) ? calendarEvents : [];
  const eventTitles = campusEvents
    .slice(0, 2)
    .map((event) => (typeof event?.title === "string" ? event.title.trim() : ""))
    .filter(Boolean);

  if (campusStatement) {
    return campusStatement;
  }

  if (campusEvents.length > 0) {
    const titlesText =
      eventTitles.length > 0
        ? `, darunter ${eventTitles.join(eventTitles.length > 1 ? " und " : "")}`
        : "";
    return `Ich sehe außerdem ${formatCountPhrase(campusEvents.length, "relevanten Termin", "relevante Termine")} im Umfeld der Bibliothek${titlesText}. Solche Termine können die Auslastung zusätzlich verschieben.`;
  }

  return "";
}

function buildScheduleNarrative({ campusContext, calendarEvents }) {
  const parts = [
    buildLibraryHoursNarrative(campusContext?.library_hours),
    buildCampusContextNarrative({ campusContext, calendarEvents }),
  ].filter(Boolean);

  return parts.join(" ");
}

function buildFallbackNarrative({
  explanation,
  forecastPeak,
  forecastCurrent,
  currentPersons,
  zoneCapacity,
  upcomingBookings,
  calendarEvents,
  campusContext,
  historyPoints,
}) {
  const scheduleNarrative = buildScheduleNarrative({ campusContext, calendarEvents });
  const trend = analyzeHistoryTrend(historyPoints);
  const trendNarrative = buildTrendNarrative(historyPoints);
  const nextBooking = Array.isArray(upcomingBookings) ? upcomingBookings[0] ?? null : null;
  const bookingNarrative = buildBookingNarrative({ upcomingBookings, nextBooking });
  const libraryHours = campusContext?.library_hours ?? null;
  const capacityPressureNarrative = buildCapacityPressureNarrative({
    currentPersons,
    zoneCapacity,
    forecastCurrent,
    forecastPeak,
  });
  const drivers = Array.isArray(explanation?.drivers) ? explanation.drivers.slice(0, 4) : [];
  const uncertaintyLevel = explanation?.uncertainty?.level ?? "unbekannt";
  const bookingCount = upcomingBookings.length;
  const eventCount = calendarEvents.length;
  const suppressedDriverKeys = new Set();

  if (scheduleNarrative) {
    ["lecture_density", "lecture_start_wave", "event_context"].forEach((key) => suppressedDriverKeys.add(key));
  }
  if (trendNarrative) {
    ["trend", "momentum"].forEach((key) => suppressedDriverKeys.add(key));
  }
  if (bookingNarrative) {
    ["buchungen", "bookings"].forEach((key) => suppressedDriverKeys.add(key));
  }

  const interpretedDrivers = drivers
    .filter((driver) => !suppressedDriverKeys.has(normalizeDriverKey(driver?.name)))
    .map((driver) => buildDriverInterpretation(driver, { bookingCount, eventCount }))
    .filter(Boolean);
  const occupancyDelta =
    typeof forecastCurrent === "number" && typeof currentPersons === "number"
      ? forecastCurrent - currentPersons
      : null;

  let intro = "";
  if (libraryHours?.is_open_now === false) {
    intro = "Kurz gesagt: Laut den Öffnungszeiten ist die Bibliothek aktuell geschlossen.";
    if (typeof forecastCurrent === "number" || typeof forecastPeak === "number") {
      intro += ` Das Rohmodell liegt zwar rechnerisch bei etwa ${formatWholeNumber(forecastCurrent)} bis ${formatWholeNumber(forecastPeak)} Personen${zoneCapacity ? ` bei ${formatWholeNumber(zoneCapacity)} verfügbaren Plätzen` : ""}, für Besucher ist die Schließung aber der klar dominierende Faktor.`;
    } else {
      intro += " Für den kurzfristigen Ausblick ist deshalb eher keine zusätzliche Nachfrage bis zur Wiederöffnung zu erwarten.";
    }
  } else if (libraryHours?.closes_within_horizon && typeof libraryHours?.today_close === "string") {
    intro = `Kurz gesagt: Aktuell sind ungefähr ${formatWholeNumber(currentPersons)} Personen in der Bibliothek. Die Bibliothek schließt heute um ${libraryHours.today_close} Uhr und damit innerhalb des 3-Stunden-Prognosefensters ab +30 Minuten.`;
    if (typeof forecastCurrent === "number" || typeof forecastPeak === "number") {
      intro += ` Selbst wenn das Rohmodell rechnerisch noch etwa ${formatWholeNumber(forecastCurrent)} bis ${formatWholeNumber(forecastPeak)} Personen sieht${zoneCapacity ? `, bei insgesamt ${formatWholeNumber(zoneCapacity)} verfügbaren Plätzen` : ""}, begrenzt die anstehende Schließung spätere Nachfrage deutlich.`;
    }
  } else {
    intro = `Kurz gesagt: Aktuell sind ungefähr ${formatWholeNumber(currentPersons)} Personen in der Bibliothek.`;
    if (typeof occupancyDelta === "number") {
      if (occupancyDelta > 10) {
        intro += ` Im 3-Stunden-Fenster ab +30 Minuten rechne ich damit, dass es deutlich voller wird und sich die Auslastung in Richtung ${formatWholeNumber(forecastCurrent)} bis ${formatWholeNumber(forecastPeak)} Personen bewegt${zoneCapacity ? `, bei insgesamt ${formatWholeNumber(zoneCapacity)} verfügbaren Plätzen` : ""}.`;
      } else if (occupancyDelta < -10) {
        intro += ` Im 3-Stunden-Fenster ab +30 Minuten erwarte ich eher eine Entspannung und damit weniger Belegung als im aktuellen Moment. Der wahrscheinliche Bereich liegt bei etwa ${formatWholeNumber(forecastCurrent)} bis ${formatWholeNumber(forecastPeak)} Personen${zoneCapacity ? ` bei einer Kapazität von ${formatWholeNumber(zoneCapacity)} Plätzen` : ""}.`;
      } else {
        intro += ` Für das 3-Stunden-Fenster ab +30 Minuten erwarte ich keine extreme Verschiebung, sondern eher einen Bereich um ${formatWholeNumber(forecastCurrent)} bis ${formatWholeNumber(forecastPeak)} Personen${zoneCapacity ? ` bei einer Kapazität von ${formatWholeNumber(zoneCapacity)} Plätzen` : ""}.`;
      }
    } else {
      intro += ` Für das 3-Stunden-Fenster ab +30 Minuten liegt die erwartete Auslastung ungefähr im Bereich von ${formatWholeNumber(forecastCurrent)} bis ${formatWholeNumber(forecastPeak)} Personen${zoneCapacity ? ` bei einer Kapazität von ${formatWholeNumber(zoneCapacity)} Plätzen` : ""}.`;
    }
  }

  const whyParts = [
    capacityPressureNarrative,
    trendNarrative,
    scheduleNarrative,
    bookingNarrative,
    ...interpretedDrivers,
  ].filter(Boolean);
  const why =
    whyParts.length > 0
      ? `Das liegt vor allem an Folgendem: ${whyParts.join(" ")}`
      : "Diese Einschätzung leite ich vor allem aus dem aktuellen Belegungsverlauf, den campusbezogenen Einflussfaktoren und bereits bekannten Nachfrageimpulsen ab.";
  const uncertaintyText = buildUncertaintyNarrative(uncertaintyLevel, drivers);
  const recommendationText = buildRecommendationNarrative({
    forecastPeak,
    zoneCapacity,
    freeSeats:
      typeof zoneCapacity === "number" && typeof currentPersons === "number"
        ? Math.max(zoneCapacity - currentPersons, 0)
        : null,
    uncertaintyLevel,
    upcomingBookings,
    trendLabel: trend.label,
    libraryHours,
  });

  return [intro, why, `Wichtig dabei ist noch: ${uncertaintyText}`, recommendationText]
    .filter(Boolean)
    .join("\n\n");
}

function buildNarrativePrompt({
  currentPersons,
  zoneCapacity,
  forecastCurrent,
  forecastPeak,
  upcomingBookings,
  calendarEvents,
  campusContext,
  historyPoints,
}) {
  const parts = [
    "Erkläre für Studierende in einfacher Sprache, warum die Bibliothek ab 30 Minuten nach der aktuellen Uhrzeit für die folgenden 3 Stunden so prognostiziert wird.",
  ];
  const current = asNumber(currentPersons);
  const capacity = asNumber(zoneCapacity);
  const forecastValue = asNumber(forecastCurrent);
  const peakValue = asNumber(forecastPeak);
  const bookingCount = Array.isArray(upcomingBookings) ? upcomingBookings.length : 0;
  const eventCount = Array.isArray(calendarEvents) ? calendarEvents.length : 0;
  const libraryHours = campusContext?.library_hours ?? null;
  const activeCampusLectures = Math.max(0, Math.round(asNumber(campusContext?.active_onsite_lectures) ?? 0));
  const campusStartsNextHour = Math.max(0, Math.round(asNumber(campusContext?.starting_next_60m) ?? 0));
  const campusEndsNextHour = Math.max(0, Math.round(asNumber(campusContext?.ending_next_60m) ?? 0));
  const trend = analyzeHistoryTrend(historyPoints);

  if (current !== null) {
    parts.push(`Aktuell sind ungefähr ${Math.round(current)} Personen vor Ort.`);
  }

  if (current !== null && capacity !== null && capacity > 0) {
    parts.push(`Das entspricht rund ${Math.round((current / capacity) * 100)} Prozent der Kapazität von ${Math.round(capacity)} Plätzen.`);
  }

  if (forecastValue !== null) {
    parts.push(`Der erste Prognosepunkt in +30 Minuten liegt bei etwa ${Math.round(forecastValue)} Personen.`);
  }

  if (peakValue !== null) {
    parts.push(`Der erwartete Peak liegt bei ungefähr ${Math.round(peakValue)} Personen.`);
  }

  if (trend.label === "steigend") {
    parts.push("Der jüngste Live-Trend zeigt nach oben.");
  } else if (trend.label === "fallend") {
    parts.push("Der jüngste Live-Trend zeigt eher nach unten.");
  } else if (trend.label === "stabil") {
    parts.push("Der jüngste Live-Trend wirkt eher stabil.");
  }

  if (bookingCount > 0) {
    parts.push(`Es gibt ${bookingCount} bestätigte Buchungen in der näheren Zukunft.`);
  }

  if (libraryHours?.is_open_now === false) {
    parts.push(`Die Bibliothek ist laut Öffnungszeiten aktuell geschlossen${typeof libraryHours?.next_open_label === "string" ? ` und öffnet wieder ${libraryHours.next_open_label}.` : "."}`);
  } else if (typeof libraryHours?.today_close === "string") {
    parts.push(`Die Bibliothek ist heute bis ${libraryHours.today_close} Uhr geöffnet.`);
    if (libraryHours?.closes_within_horizon && typeof libraryHours?.minutes_until_close === "number") {
      parts.push(`Die Schließung liegt damit in etwa ${libraryHours.minutes_until_close} Minuten innerhalb des Prognosehorizonts.`);
    }
  }

  if (activeCampusLectures > 0) {
    parts.push(`Laut dhbw.app laufen am Standort Mannheim aktuell ungefähr ${activeCampusLectures} Präsenzveranstaltungen.`);
  }

  if (campusStartsNextHour > 0 || campusEndsNextHour > 0) {
    parts.push(`Für Mannheim zeigt der öffentliche Stundenplan in der nächsten Stunde ${campusStartsNextHour} Starts und ${campusEndsNextHour} Enden von Präsenzveranstaltungen.`);
  } else if (eventCount > 0) {
    parts.push(`Außerdem gibt es ${eventCount} relevante Campus-Termine.`);
  }

  parts.push("Wenn die Bibliothek vor Ablauf des 3-Stunden-Prognosefensters ab +30 Minuten schließt, behandle das als dominanten begrenzenden Faktor.");
  parts.push("Nenne die zwei bis drei wichtigsten Treiber, die Unsicherheit und was das konkret für Besucher bedeutet.");
  return parts.join(" ");
}

function extractNarrativeText(payload) {
  const narrative = payload?.response?.narrative ?? {};
  const parts = [
    narrative?.one_liner,
    narrative?.warum,
    narrative?.unsicherheit,
    narrative?.empfehlung,
  ].filter((value) => typeof value === "string" && value.trim());

  if (parts.length > 0) {
    return postProcessNarrativeText(parts.join(" "));
  }

  if (typeof payload?.narrative_markdown === "string" && payload.narrative_markdown.trim()) {
    return postProcessNarrativeText(payload.narrative_markdown
      .replace(/^#+\s*/gm, "")
      .replace(/^\s*[-*]\s+/gm, "")
      .replace(/\n{2,}/g, "\n")
      .trim());
  }

  return "";
}

export default function HomePage() {
  const {
    authenticated,
    user,
    bookings,
    bookingRevision,
    sessionLoading,
  } = useAppData();

  const [zoneCapacity, setZoneCapacity] = useState(null);
  const [zonesError, setZonesError] = useState("");
  const [hubOverview, setHubOverview] = useState(null);
  const [hubError, setHubError] = useState("");
  const [commandCenter, setCommandCenter] = useState(null);
  const [commandCenterError, setCommandCenterError] = useState("");
  const [campusContext, setCampusContext] = useState(null);
  const [narrativeText, setNarrativeText] = useState("");
  const [narrativeSource, setNarrativeSource] = useState("loading");
  const [narrativeError, setNarrativeError] = useState("");
  const hubFailureCountRef = useRef(0);
  const commandCenterFailureCountRef = useRef(0);
  const narrativeInFlightRef = useRef(false);
  const narrativePromptRef = useRef(
    "Erkläre für Studierende in einfacher Sprache, warum die Bibliothek ab 30 Minuten nach der aktuellen Uhrzeit für die folgenden 3 Stunden so prognostiziert wird.",
  );

  useEffect(() => {
    let active = true;
    const abortController = new AbortController();

    async function loadZones() {
      try {
        const zones = await sitcheckApi.getZones(abortController.signal);
        if (!active) return;
        const defaultZone = Array.isArray(zones)
          ? zones.find((zone) => zone.zone_id === sitcheckApi.defaultZoneId)
          : null;
        setZoneCapacity(defaultZone?.capacity ?? null);
        setZonesError("");
      } catch (error) {
        if (!active) return;
        const message = error instanceof Error ? error.message : "Zonen konnten nicht geladen werden.";
        setZonesError(message);
      }
    }

    loadZones();
    return () => {
      active = false;
      abortController.abort();
    };
  }, []);

  useEffect(() => {
    let active = true;

    async function loadHub() {
      try {
        const payload = await sitcheckApi.getHubOverview();
        if (!active) return;
        hubFailureCountRef.current = 0;
        setHubOverview(payload);
        setHubError("");
      } catch (error) {
        if (!active) return;
        hubFailureCountRef.current += 1;
        if (hubFailureCountRef.current >= 2) {
          const message = error instanceof Error ? error.message : "Hub-Übersicht konnte nicht geladen werden.";
          setHubError(message);
        }
      }
    }

    loadHub();
    const intervalId = window.setInterval(loadHub, 5000);
    return () => {
      active = false;
      window.clearInterval(intervalId);
    };
  }, []);

  useEffect(() => {
    let active = true;

    async function loadNarrative() {
      if (narrativeInFlightRef.current) {
        return;
      }
      narrativeInFlightRef.current = true;
      setNarrativeSource((currentSource) => (currentSource === "llm" ? currentSource : "loading"));
      setNarrativeError("");
      const abortController = new AbortController();
      const timeoutId = window.setTimeout(() => abortController.abort(), NARRATIVE_TIMEOUT_MS);

      try {
        const payload = await sitcheckApi.getExplainNarrative({
          signal: abortController.signal,
          query: narrativePromptRef.current,
        });
        if (!active) return;

        const extractedText = extractNarrativeText(payload);
        if (extractedText) {
          setNarrativeText(extractedText);
          setNarrativeSource("llm");
          setNarrativeError("");
          return;
        }
      } catch (error) {
        if (!active) return;
        const message =
          error instanceof Error && error.name === "AbortError"
            ? "Erklärtext-Timeout"
            : error instanceof Error
            ? error.message
            : "";
        setNarrativeError(message);
      } finally {
        narrativeInFlightRef.current = false;
        window.clearTimeout(timeoutId);
      }

      if (!active) return;
      setNarrativeText("");
      setNarrativeSource("fallback");
    }

    async function loadCommandCenter() {
      try {
        const payload = await sitcheckApi.getCommandCenter();
        if (!active) return;
        commandCenterFailureCountRef.current = 0;
        setCommandCenter(payload);
        setCommandCenterError("");
      } catch (error) {
        if (!active) return;
        commandCenterFailureCountRef.current += 1;
        if (commandCenterFailureCountRef.current >= 2) {
          const message =
            error instanceof Error
              ? error.message
              : "Command Center konnte nicht geladen werden.";
          setCommandCenterError(message);
        }
      }
    }

    async function loadCampusContext() {
      try {
        const payload = await sitcheckApi.getDhbwMannheimContext();
        if (!active) return;
        setCampusContext(payload);
      } catch {
        if (!active) return;
        setCampusContext(null);
      }
    }

    loadCommandCenter().catch(() => {});
    loadNarrative().catch(() => {});
    loadCampusContext().catch(() => {});
    const intervalId = window.setInterval(loadCommandCenter, 60000);
    const narrativeIntervalId = window.setInterval(loadNarrative, 60000);
    const campusContextIntervalId = window.setInterval(loadCampusContext, 60000);
    return () => {
      active = false;
      window.clearInterval(intervalId);
      window.clearInterval(narrativeIntervalId);
      window.clearInterval(campusContextIntervalId);
    };
  }, [bookingRevision]);

  const occupancy = hubOverview?.occupancy ?? {};
  const currentPersons = typeof occupancy.currentPersons === "number" ? occupancy.currentPersons : 0;
  const averagePersons = typeof occupancy.averagePersons === "number" ? occupancy.averagePersons : null;
  const utilization = zoneCapacity ? Math.round((currentPersons / zoneCapacity) * 100) : null;
  const freeSeats = zoneCapacity ? Math.max(zoneCapacity - currentPersons, 0) : null;

  const commandCenterHistoryPoints = Array.isArray(commandCenter?.history?.points)
    ? commandCenter.history.points
    : [];
  const hubHistoryPoints = Array.isArray(occupancy?.history) ? occupancy.history : [];
  const resolvedHistoryPoints =
    commandCenterHistoryPoints.length > 0 ? commandCenterHistoryPoints : hubHistoryPoints;
  const historyChartData = resolvedHistoryPoints.slice(-24).map((point) => ({
    label: formatTime(point.timestamp),
    occupancy:
      typeof point.occupancy === "number"
        ? point.occupancy
        : typeof point.persons === "number"
          ? point.persons
          : 0,
  }));
  const historySourceLabel =
    commandCenterHistoryPoints.length > 0
      ? "Command Center"
      : hubHistoryPoints.length > 0
        ? "Hub-Fallback"
        : null;

  const commandCenterForecastPoints = Array.isArray(commandCenter?.forecast_latest?.points)
    ? commandCenter.forecast_latest.points
    : [];
  const hubForecastPoints = Array.isArray(hubOverview?.forecast?.points)
    ? hubOverview.forecast.points
    : [];
  const forecastPoints =
    commandCenterForecastPoints.length > 0 ? commandCenterForecastPoints : hubForecastPoints;
  const forecastChartData = buildForecastChartData(forecastPoints);

  const forecastPeak = forecastChartData.length > 0
    ? forecastChartData.reduce((peak, point) => {
        const forecast = asNumber(point?.forecast);
        return forecast !== null && forecast > peak ? forecast : peak;
      }, 0)
    : null;
  const forecastCurrent = forecastChartData.length > 0 ? forecastChartData[0].forecast : null;

  const explanation = commandCenter?.explanation ?? {};
  const calendarEvents = Array.isArray(commandCenter?.calendar_events) ? commandCenter.calendar_events : [];

  const upcomingBookings = bookings
    .filter((booking) => booking.status === "confirmed" && new Date(booking.ends_at) >= new Date())
    .sort((left, right) => new Date(left.starts_at).getTime() - new Date(right.starts_at).getTime());
  const nextBooking = upcomingBookings[0] ?? null;
  const liveExplainabilitySnapshot = buildLiveExplainabilitySnapshot({
    currentPersons,
    zoneCapacity,
    forecastCurrent,
    forecastPeak,
    upcomingBookings,
    calendarEvents,
    campusContext,
    historyPoints: resolvedHistoryPoints,
  });
  const narrativePrompt = buildNarrativePrompt({
    currentPersons,
    zoneCapacity,
    forecastCurrent,
    forecastPeak,
    upcomingBookings,
    calendarEvents,
    campusContext,
    historyPoints: resolvedHistoryPoints,
  });
  narrativePromptRef.current = narrativePrompt;
  const fallbackNarrativeText = buildFallbackNarrative({
    explanation,
    forecastPeak,
    forecastCurrent,
    currentPersons,
    zoneCapacity,
    upcomingBookings,
    calendarEvents,
    campusContext,
    historyPoints: resolvedHistoryPoints,
  });
  const libraryHoursNarrative = buildLibraryHoursNarrative(campusContext?.library_hours);
  const campusNarrative = buildCampusContextNarrative({
    campusContext,
    calendarEvents,
  });
  const llmSupplementParts = [];

  if (libraryHoursNarrative && !/schlie|öffnungszeit|geschlossen|öffnet|geöffnet|offen/i.test(narrativeText)) {
    llmSupplementParts.push(`Öffnungszeiten: ${libraryHoursNarrative}`);
  }

  if (campusNarrative && !/vorles|kalender|termin|campusgeschehen|veranstaltung|dhbw\.app|stundenplan/i.test(narrativeText)) {
    llmSupplementParts.push(`Campus-Kontext: ${campusNarrative}`);
  }

  const llmNarrative = narrativeText
    ? [narrativeText, ...llmSupplementParts].filter(Boolean).join("\n\n")
    : narrativeText;
  const readableNarrative = llmNarrative
    || (narrativeSource === "fallback"
      ? fallbackNarrativeText
      : "Qwen erstellt gerade eine frische Erklärung aus Live-Daten, Prognose und Campusgeschehen. Der Text erscheint, sobald das lokale Modell fertig ist.");
  const narrativeStatus =
    !llmNarrative && narrativeSource === "loading"
      ? "Lokales Qwen-Modell wird gerade angefragt."
      : !llmNarrative && narrativeSource === "fallback"
      ? narrativeError
        ? `Lokales Qwen-Modell war nicht verfügbar (${narrativeError}). Deshalb wird vorübergehend eine heuristische Kurzfassung angezeigt.`
        : "Lokales Qwen-Modell war nicht verfügbar. Deshalb wird vorübergehend eine heuristische Kurzfassung angezeigt."
      : "";

  const liveCards = [
    { label: "Personen live", value: currentPersons, hint: "aktueller Zählerstand" },
    { label: "Kapazität", value: zoneCapacity ?? "–", hint: "aktive Bibliotheksentität" },
    { label: "Freie Plätze", value: freeSeats ?? "–", hint: "rein rechnerisch verfügbar" },
    { label: "Auslastung", value: utilization !== null ? `${utilization}%` : "–", hint: "bezogen auf die Bibliothekskapazität" },
    { label: "3h Peak", value: forecastPeak !== null ? Math.round(forecastPeak) : "–", hint: "Spitze ab +30 min inkl. Buchungs-Overlay" },
    { label: "Durchschnitt", value: averagePersons !== null ? averagePersons.toFixed(1) : "–", hint: "historischer Mittelwert im Hub" },
  ];

  return (
    <div className="flex flex-col gap-8">
      <section className="overflow-hidden rounded-[2rem] border border-white/70 bg-[linear-gradient(135deg,rgba(72,123,163,0.96),rgba(28,54,78,0.95))] p-6 text-white shadow-[0_30px_80px_rgba(39,72,102,0.22)] sm:p-8">
	        <div className="flex flex-col gap-8 lg:flex-row lg:items-center lg:justify-between">
          <div className="max-w-3xl space-y-4">
            <span className="inline-flex w-fit items-center rounded-full border border-white/15 bg-white/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.22em] text-blue-100">
              Smarter finden, Besser lernen
            </span>
            <div className="space-y-2">
              <h2 className="text-3xl font-semibold sm:text-4xl">Live-Lagebild des Learning Centers DHBW</h2>
              <p className="max-w-2xl text-sm leading-6 text-slate-200 sm:text-base">
                Sitcheck bündelt Live-Auslastung, 3-Stunden-Prognose ab +30 Minuten, Explainable AI und
                Nutzerbuchungen für die gesamte Bibliothek in einer einzigen Oberfläche.
              </p>
            </div>
          </div>

          <div className="grid gap-3 sm:min-w-[20rem]">
            {authenticated ? (
              <div className="rounded-[1.75rem] border border-white/15 bg-white/10 p-5">
                <p className="text-xs font-semibold uppercase tracking-[0.22em] text-blue-100">
                  Aktiver Account
                </p>
                <p className="mt-2 text-2xl font-semibold">{user?.username}</p>
                <p className="mt-1 text-sm text-slate-200">
                  Buchungen werden direkt im Backend gespeichert und im Forecast berücksichtigt.
                </p>
                <Link
                  href="/bookings"
                  className="mt-4 inline-flex items-center gap-2 rounded-full bg-white px-4 py-2 text-sm font-semibold text-[color:var(--accent-strong)] transition hover:bg-cyan-50"
                >
                  Buchungen verwalten
                  <span aria-hidden="true">→</span>
                </Link>
              </div>
            ) : (
              <div className="rounded-[1.75rem] border border-white/15 bg-white/10 p-5">
                <p className="text-xs font-semibold uppercase tracking-[0.22em] text-blue-100">
                  Login für Buchungen
                </p>
                <p className="mt-2 text-sm text-slate-200">
                  Ohne Konto siehst du Live-Daten und Prognosen. Für Reservierungen und den
                  Einfluss auf die Prognose ist ein Login erforderlich.
                </p>
                <Link
                  href="/login"
                  className="mt-4 inline-flex items-center gap-2 rounded-full bg-white px-4 py-2 text-sm font-semibold text-[color:var(--accent-strong)] transition hover:bg-cyan-50"
                >
                  Jetzt anmelden
                  <span aria-hidden="true">→</span>
                </Link>
              </div>
            )}
          </div>
        </div>

        {(hubError || commandCenterError || zonesError) && (
          <div className="mt-6 grid gap-3 lg:grid-cols-3">
            {hubError && (
              <div className="rounded-2xl border border-amber-200/35 bg-amber-200/15 px-4 py-3 text-sm text-amber-50">
                Hub: {hubError}
              </div>
            )}
            {commandCenterError && (
              <div className="rounded-2xl border border-amber-200/35 bg-amber-200/15 px-4 py-3 text-sm text-amber-50">
                Command Center: {commandCenterError}
              </div>
            )}
            {zonesError && (
              <div className="rounded-2xl border border-amber-200/35 bg-amber-200/15 px-4 py-3 text-sm text-amber-50">
                Zonen: {zonesError}
              </div>
            )}
          </div>
        )}
      </section>

      <section className="grid gap-4 lg:grid-cols-6">
        {liveCards.map((card) => (
          <article
            key={card.label}
            className="rounded-[1.75rem] border border-[color:var(--stroke-soft)] bg-[color:var(--surface-raised)] p-5 shadow-[0_24px_54px_rgba(15,23,42,0.08)]"
          >
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--ink-muted)]">
              {card.label}
            </p>
            <p className="mt-3 text-3xl font-semibold text-[color:var(--ink-strong)]">
              {card.value}
            </p>
            <p className="mt-2 text-sm leading-6 text-[color:var(--ink-soft)]">{card.hint}</p>
          </article>
        ))}
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.4fr_1fr]">
        <article className="rounded-[2rem] border border-[color:var(--stroke-soft)] bg-[color:var(--surface-raised)] p-6 shadow-[0_24px_54px_rgba(15,23,42,0.08)]">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[color:var(--ink-muted)]">
                3-Stunden-Prognose
              </p>
              <h3 className="mt-1 text-2xl font-semibold text-[color:var(--ink-strong)]">
                Prognoseband inkl. Buchungs-Overlay
              </h3>
              <p className="mt-2 text-sm leading-6 text-[color:var(--ink-soft)]">
                Die Darstellung startet {FORECAST_WINDOW_LABEL}, aggregiert die Messpunkte im
                10-Minuten-Takt und zeigt die Prognose inkl. Buchungs-Overlay.
              </p>
            </div>
            <div className="rounded-2xl bg-[color:var(--surface-muted)] px-4 py-3 text-sm text-[color:var(--ink-strong)]">
              <p>Erster Wert (+30 min): {forecastCurrent ?? "–"}</p>
              <p>Peak: {forecastPeak !== null ? Math.round(forecastPeak) : "–"}</p>
            </div>
          </div>

          <div className="mt-6 h-80">
            {forecastChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={forecastChartData} margin={{ top: 12, right: 20, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="forecastFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#8bc6ec" stopOpacity={0.38} />
                      <stop offset="95%" stopColor="#8bc6ec" stopOpacity={0.05} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#d8e3e1" />
                  <XAxis dataKey="label" stroke="#6e8381" />
                  <YAxis allowDecimals={false} stroke="#6e8381" />
                  <Tooltip
                    formatter={(value, name) => {
                      const labels = {
                        forecast: "Forecast",
                        lower: "Unteres Band",
                        upper: "Oberes Band",
                      };
                      return [`${value} Personen`, labels[name] ?? name];
                    }}
                  />
                  <Area type="monotone" dataKey="upper" stroke="#9fd3f2" fill="url(#forecastFill)" />
                  <Area type="monotone" dataKey="lower" stroke="#d9edf9" fillOpacity={0} />
                  <Line type="monotone" dataKey="forecast" stroke="#487ba3" strokeWidth={3} dot={{ r: 3 }} />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-full items-center justify-center rounded-[1.5rem] border border-dashed border-[color:var(--stroke-strong)] bg-[color:var(--surface-muted)] px-6 text-center text-sm text-[color:var(--ink-soft)]">
                Noch keine Forecast-Punkte verfügbar.
              </div>
            )}
          </div>
        </article>

        <article className="rounded-[2rem] border border-[color:var(--stroke-soft)] bg-[color:var(--surface-raised)] p-6 shadow-[0_24px_54px_rgba(15,23,42,0.08)]">
          <h3 className="text-2xl font-semibold text-[color:var(--ink-strong)]">
            Explainable AI
          </h3>
          <div className="mt-3 rounded-[1.5rem] border border-[color:var(--stroke-soft)] bg-[linear-gradient(180deg,rgba(255,255,255,0.92),rgba(241,247,246,0.96))] px-5 py-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.7)]">
            <p className="whitespace-pre-line text-[15px] leading-8 text-[color:var(--ink-strong)]">
              {readableNarrative}
            </p>
            {narrativeStatus && (
              <p className="mt-4 rounded-[1rem] border border-[color:var(--stroke-soft)] bg-white/80 px-4 py-3 text-sm leading-7 text-[color:var(--ink-soft)]">
                {narrativeStatus}
              </p>
            )}
            {liveExplainabilitySnapshot && (
              <p className="mt-4 rounded-[1rem] border border-[color:var(--stroke-soft)] bg-white/80 px-4 py-3 text-sm leading-7 text-[color:var(--ink-soft)]">
                {liveExplainabilitySnapshot}
              </p>
            )}
          </div>
        </article>
      </section>

      <section className="grid gap-6">
        <article className="rounded-[2rem] border border-[color:var(--stroke-soft)] bg-[color:var(--surface-raised)] p-6 shadow-[0_24px_54px_rgba(15,23,42,0.08)]">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[color:var(--ink-muted)]">
                Live-Historie
              </p>
              <h3 className="mt-1 text-2xl font-semibold text-[color:var(--ink-strong)]">
                Verlauf der letzten Messpunkte
              </h3>
            </div>
            <p className="text-sm text-[color:var(--ink-soft)]">
              Letztes Live-Update: {formatDateTime(commandCenter?.live?.timestamp ?? occupancy?.lastUpdated)}
              {historySourceLabel ? ` • Quelle: ${historySourceLabel}` : ""}
            </p>
          </div>

          <div className="mt-6 h-72">
            {historyChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={historyChartData} margin={{ top: 12, right: 20, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#d8e3e1" />
                  <XAxis dataKey="label" stroke="#6e8381" />
                  <YAxis allowDecimals={false} stroke="#6e8381" />
                  <Tooltip formatter={(value) => [`${value} Personen`, "Live-Wert"]} />
                  <Line type="monotone" dataKey="occupancy" stroke="#5a96c0" strokeWidth={3} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-full items-center justify-center rounded-[1.5rem] border border-dashed border-[color:var(--stroke-strong)] bg-[color:var(--surface-muted)] px-6 text-center text-sm text-[color:var(--ink-soft)]">
                Noch keine historischen Datenpunkte vorhanden.
              </div>
            )}
          </div>
        </article>
      </section>

      <section className="grid gap-6">
        <article className="rounded-[2rem] border border-[color:var(--stroke-soft)] bg-[color:var(--surface-raised)] p-6 shadow-[0_24px_54px_rgba(15,23,42,0.08)]">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[color:var(--ink-muted)]">
            Buchungsstatus
          </p>
          <h3 className="mt-1 text-2xl font-semibold text-[color:var(--ink-strong)]">
            Einfluss deiner Reservierungen
          </h3>
          {sessionLoading ? (
            <p className="mt-4 text-sm text-[color:var(--ink-soft)]">Sitzung wird geladen …</p>
          ) : authenticated ? (
            <div className="mt-4 space-y-4">
              <p className="text-sm leading-6 text-[color:var(--ink-soft)]">
                Alle bestätigten Bibliotheks-Buchungen aller Nutzer zählen im MVP als bekannte
                Nachfrage und fließen gemeinsam in die 3-Stunden-Prognose ab +30 Minuten ein. Deine Buchungen
                sind also nur ein Teil des globalen Effekts.
              </p>
              <div className="rounded-[1.5rem] bg-[color:var(--surface-muted)] px-4 py-4">
                <p className="text-sm font-semibold text-[color:var(--ink-strong)]">
                  {nextBooking ? "Nächste Buchung" : "Noch keine aktive Buchung"}
                </p>
                {nextBooking ? (
                  <div className="mt-2 text-sm leading-6 text-[color:var(--ink-soft)]">
                    <p>Start: {formatDateTime(nextBooking.starts_at)}</p>
                    <p>Ende: {formatDateTime(nextBooking.ends_at)}</p>
                    <p>Status: {nextBooking.status}</p>
                  </div>
                ) : (
                  <p className="mt-2 text-sm leading-6 text-[color:var(--ink-soft)]">
                    Lege jetzt ein Zeitfenster an. Deine Buchung wird gespeichert und geht danach
                    zusammen mit allen anderen bestätigten Buchungen in die Prognose ein.
                  </p>
                )}
              </div>
              <Link
                href="/bookings"
                className="inline-flex items-center gap-2 rounded-full bg-[color:var(--accent)] px-4 py-2 text-sm font-semibold text-white transition hover:bg-[color:var(--accent-strong)]"
              >
                Buchungen öffnen
                <span aria-hidden="true">→</span>
              </Link>
            </div>
          ) : (
            <div className="mt-4 space-y-4">
              <p className="text-sm leading-6 text-[color:var(--ink-soft)]">
                Für persönliche Buchungen wird ein Konto benötigt. In die Prognose fließen
                anschließend immer alle bestätigten Buchungen aller Nutzer ein.
              </p>
              <Link
                href="/login"
                className="inline-flex items-center gap-2 rounded-full bg-[color:var(--accent)] px-4 py-2 text-sm font-semibold text-white transition hover:bg-[color:var(--accent-strong)]"
              >
                Login oder Registrierung
                <span aria-hidden="true">→</span>
              </Link>
            </div>
          )}
        </article>
      </section>
    </div>
  );
}
