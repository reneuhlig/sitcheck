"use client";

import { useState } from "react";
import Link from "next/link";
import { useAppData } from "@/context/AppDataContext";

const HOUR_OPTIONS = Array.from({ length: 24 }, (_, index) => String(index).padStart(2, "0"));
const MINUTE_OPTIONS = Array.from({ length: 60 }, (_, index) => String(index).padStart(2, "0"));
const LIBRARY_HOURS_LABEL = "Mo-Fr 10:00-22:00, Sa-So 10:00-18:00";
const BOOKING_ERROR_MESSAGES = {
  booking_must_start_in_future: "Der Startzeitpunkt muss in der Zukunft liegen.",
  booking_end_before_start: "Das Ende muss nach dem Start liegen.",
  booking_must_stay_within_single_open_day:
    "Buchungen müssen innerhalb eines Öffnungstags liegen und können nicht über Mitternacht gehen.",
  booking_outside_opening_hours: `Buchungen sind nur während der Öffnungszeiten möglich: ${LIBRARY_HOURS_LABEL}.`,
};

function toLocalDateTimeValue(date) {
  const localDate = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
  return localDate.toISOString().slice(0, 16);
}

function toLocalDateValue(date) {
  return toLocalDateTimeValue(date).slice(0, 10);
}

function toLocalTimeValue(date) {
  return toLocalDateTimeValue(date).slice(11, 16);
}

function getOpeningHoursForDate(date) {
  const isWeekend = date.getDay() === 0 || date.getDay() === 6;
  return {
    openHour: 10,
    closeHour: isWeekend ? 18 : 22,
  };
}

function setLocalClock(date, hour, minute = 0) {
  const next = new Date(date);
  next.setHours(hour, minute, 0, 0);
  return next;
}

function alignToOpeningHours(date) {
  const candidate = new Date(date);
  candidate.setSeconds(0, 0);

  while (true) {
    const { openHour, closeHour } = getOpeningHoursForDate(candidate);
    const opensAt = setLocalClock(candidate, openHour, 0);
    const closesAt = setLocalClock(candidate, closeHour, 0);

    if (candidate < opensAt) {
      return opensAt;
    }
    if (candidate >= closesAt) {
      candidate.setDate(candidate.getDate() + 1);
      candidate.setHours(0, 0, 0, 0);
      continue;
    }
    return candidate;
  }
}

function nextBookableStart(leadMinutes = 1) {
  const candidate = new Date();
  candidate.setSeconds(0, 0);
  candidate.setMinutes(candidate.getMinutes() + leadMinutes);
  return alignToOpeningHours(candidate);
}

function getClosingDate(date) {
  const { closeHour } = getOpeningHoursForDate(date);
  return setLocalClock(date, closeHour, 0);
}

function buildDefaultFormState() {
  const start = nextBookableStart(30);
  const tentativeEnd = new Date(start);
  tentativeEnd.setMinutes(tentativeEnd.getMinutes() + 60);
  const closingDate = getClosingDate(start);
  const end = tentativeEnd > closingDate ? closingDate : tentativeEnd;

  return {
    startDate: toLocalDateValue(start),
    startTime: toLocalTimeValue(start),
    endDate: toLocalDateValue(end),
    endTime: toLocalTimeValue(end),
  };
}

function combineLocalDateTime(dateValue, timeValue) {
  if (!dateValue || !timeValue) {
    return null;
  }

  const combined = new Date(`${dateValue}T${timeValue}`);
  if (Number.isNaN(combined.getTime())) {
    return null;
  }

  return combined;
}

function splitTimeValue(value) {
  const [hour = "00", minute = "00"] = typeof value === "string" ? value.split(":") : [];
  return { hour, minute };
}

function replaceTimePart(value, part, nextPartValue) {
  const { hour, minute } = splitTimeValue(value);
  return `${part === "hour" ? nextPartValue : hour}:${part === "minute" ? nextPartValue : minute}`;
}

function getOpeningWindowForDateValue(dateValue) {
  if (!dateValue) {
    return null;
  }

  const probe = new Date(`${dateValue}T12:00`);
  if (Number.isNaN(probe.getTime())) {
    return null;
  }

  const { openHour, closeHour } = getOpeningHoursForDate(probe);
  const opensAt = combineLocalDateTime(dateValue, `${String(openHour).padStart(2, "0")}:00`);
  const closesAt = combineLocalDateTime(dateValue, `${String(closeHour).padStart(2, "0")}:00`);
  if (!opensAt || !closesAt) {
    return null;
  }

  return { opensAt, closesAt };
}

function getOpeningHoursError({ startsAt, endsAt, startDateValue, endDateValue }) {
  if (startDateValue !== endDateValue) {
    return BOOKING_ERROR_MESSAGES.booking_must_stay_within_single_open_day;
  }

  const window = getOpeningWindowForDateValue(startDateValue);
  if (!window) {
    return "Die Öffnungszeiten für dieses Datum konnten nicht geprüft werden.";
  }

  if (startsAt < window.opensAt || endsAt > window.closesAt) {
    return BOOKING_ERROR_MESSAGES.booking_outside_opening_hours;
  }

  return "";
}

function getBookingErrorMessage(error) {
  const rawMessage = error instanceof Error ? error.message : "";
  if (!rawMessage) {
    return "Buchung konnte nicht erstellt werden.";
  }
  return BOOKING_ERROR_MESSAGES[rawMessage] ?? rawMessage;
}

function formatDateTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "–";
  return date.toLocaleString("de-DE", {
    weekday: "short",
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function BookingsPage() {
  const {
    authenticated,
    user,
    sessionLoading,
    bookings,
    bookingsLoading,
    bookingsError,
    createBooking,
    cancelBooking,
  } = useAppData();

  const [formState, setFormState] = useState(() => buildDefaultFormState());
  const [submitError, setSubmitError] = useState("");
  const [submitSuccess, setSubmitSuccess] = useState("");
  const [pending, setPending] = useState(false);
  const [cancellingId, setCancellingId] = useState("");
  const startTimeParts = splitTimeValue(formState.startTime);
  const endTimeParts = splitTimeValue(formState.endTime);

  const now = new Date();
  const minimumStart = nextBookableStart(1);
  const minimumStartDate = toLocalDateValue(minimumStart);
  const minimumStartTime = toLocalTimeValue(minimumStart);
  const minimumTimeForSelectedStartDate =
    formState.startDate === minimumStartDate ? minimumStartTime : undefined;
  const minimumTimeForSelectedEndDate =
    formState.endDate === formState.startDate ? formState.startTime : undefined;
  const upcomingBookings = bookings
    .filter((booking) => booking.status === "confirmed" && new Date(booking.ends_at) >= now)
    .sort((left, right) => new Date(left.starts_at).getTime() - new Date(right.starts_at).getTime());
  const historyBookings = bookings
    .filter((booking) => booking.status === "cancelled" || new Date(booking.ends_at) < now)
    .sort((left, right) => new Date(right.starts_at).getTime() - new Date(left.starts_at).getTime());

  async function handleSubmit(event) {
    event.preventDefault();
    setPending(true);
    setSubmitError("");
    setSubmitSuccess("");

    try {
      const startsAt = combineLocalDateTime(formState.startDate, formState.startTime);
      const endsAt = combineLocalDateTime(formState.endDate, formState.endTime);

      if (!startsAt || !endsAt) {
        throw new Error("Bitte Datum und Uhrzeit für Start und Ende vollständig angeben.");
      }
      if (startsAt < minimumStart) {
        throw new Error("Der Startzeitpunkt muss in der Zukunft liegen.");
      }
      if (endsAt <= startsAt) {
        throw new Error("Das Ende muss nach dem Start liegen.");
      }
      const openingHoursError = getOpeningHoursError({
        startsAt,
        endsAt,
        startDateValue: formState.startDate,
        endDateValue: formState.endDate,
      });
      if (openingHoursError) {
        throw new Error(openingHoursError);
      }

      await createBooking({
        starts_at: startsAt.toISOString(),
        ends_at: endsAt.toISOString(),
      });
      setSubmitSuccess("Buchung gespeichert. Das Zeitfenster fließt jetzt in die Kurzfristprognose ein.");
      setFormState(buildDefaultFormState());
    } catch (error) {
      setSubmitError(getBookingErrorMessage(error));
    } finally {
      setPending(false);
    }
  }

  async function handleCancel(bookingId) {
    setCancellingId(bookingId);
    setSubmitError("");
    setSubmitSuccess("");
    try {
      await cancelBooking(bookingId);
      setSubmitSuccess("Buchung storniert. Die Prognose wird beim nächsten Reload ohne dieses Zeitfenster berechnet.");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Buchung konnte nicht storniert werden.";
      setSubmitError(message);
    } finally {
      setCancellingId("");
    }
  }

  if (sessionLoading) {
    return (
      <div className="rounded-[2rem] border border-[color:var(--stroke-soft)] bg-[color:var(--surface-raised)] p-8 text-sm text-[color:var(--ink-soft)] shadow-[0_24px_54px_rgba(15,23,42,0.08)]">
        Sitzung wird geladen …
      </div>
    );
  }

  if (!authenticated) {
    return (
      <section className="rounded-[2rem] border border-[color:var(--stroke-soft)] bg-[color:var(--surface-raised)] p-8 shadow-[0_24px_54px_rgba(15,23,42,0.08)]">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[color:var(--ink-muted)]">
          Login erforderlich
        </p>
        <h2 className="mt-2 text-3xl font-semibold text-[color:var(--ink-strong)]">
          Buchungen werden an deinen Account gebunden
        </h2>
        <p className="mt-4 max-w-2xl text-sm leading-7 text-[color:var(--ink-soft)]">
          Für Reservierungen benötigt Sitcheck ein Konto. Nach dem Login kannst du Bibliotheks-Slots
          anlegen, einsehen und stornieren. Jede bestätigte Buchung zählt im MVP als +1 geplanter Platz
          in der 3-Stunden-Prognose ab +30 Minuten.
        </p>
        <Link
          href="/login"
          className="mt-6 inline-flex items-center gap-2 rounded-full bg-[color:var(--accent)] px-4 py-2 text-sm font-semibold text-white transition hover:bg-[color:var(--accent-strong)]"
        >
          Jetzt anmelden
          <span aria-hidden="true">→</span>
        </Link>
      </section>
    );
  }

  return (
    <div className="grid gap-6 xl:grid-cols-[0.95fr_1.05fr]">
      <section className="rounded-[2rem] border border-[color:var(--stroke-soft)] bg-[color:var(--surface-raised)] p-6 shadow-[0_24px_54px_rgba(15,23,42,0.08)]">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[color:var(--ink-muted)]">
          Neue Buchung
        </p>
        <h2 className="mt-2 text-3xl font-semibold text-[color:var(--ink-strong)]">
          Bibliotheks-Slot reservieren
        </h2>
        <p className="mt-3 text-sm leading-7 text-[color:var(--ink-soft)]">
          Angemeldet als <span className="font-semibold text-[color:var(--ink-strong)]">{user?.username}</span>.
          Das Backend speichert die Buchung sofort als bestätigt. Es gibt keinen Freigabefluss und keine
          Kapazitätsprüfung im MVP.
        </p>
        <p className="mt-3 text-sm leading-7 text-[color:var(--ink-soft)]">
          Hier siehst du nur deine eigenen Buchungen. Für die Bibliotheks-Prognose werden aber alle
          bestätigten Buchungen aller Nutzer gemeinsam berücksichtigt.
        </p>
        <p className="mt-3 text-sm leading-7 text-[color:var(--ink-soft)]">
          Datum und Uhrzeit kannst du für Start und Ende frei festlegen. Die vorgeschlagenen Werte
          dienen nur als Vorauswahl.
        </p>
        <p className="mt-3 text-sm leading-7 text-[color:var(--ink-soft)]">
          Öffnungszeiten: {LIBRARY_HOURS_LABEL}.
        </p>

        <form onSubmit={handleSubmit} className="mt-6 space-y-5">
          <div className="grid gap-4 md:grid-cols-2">
            <label className="block">
              <span className="text-xs font-semibold uppercase tracking-[0.16em] text-[color:var(--ink-muted)]">
                Startdatum
              </span>
              <input
                type="date"
                value={formState.startDate}
                min={minimumStartDate}
                onChange={(event) => setFormState((current) => ({ ...current, startDate: event.target.value }))}
                className="mt-2 w-full rounded-[1.25rem] border border-[color:var(--stroke-strong)] bg-white px-4 py-3 text-[color:var(--ink-strong)] outline-none focus:border-[color:var(--accent)]"
                required
              />
            </label>

            <label className="block">
              <span className="text-xs font-semibold uppercase tracking-[0.16em] text-[color:var(--ink-muted)]">
                Startzeit
              </span>
              <div className="mt-2 grid grid-cols-[1fr_auto_1fr] items-center gap-3">
                <select
                  value={startTimeParts.hour}
                  onChange={(event) =>
                    setFormState((current) => ({
                      ...current,
                      startTime: replaceTimePart(current.startTime, "hour", event.target.value),
                    }))
                  }
                  className="w-full rounded-[1.25rem] border border-[color:var(--stroke-strong)] bg-white px-4 py-3 text-[color:var(--ink-strong)] outline-none focus:border-[color:var(--accent)]"
                  aria-label="Startstunde"
                  required
                >
                  {HOUR_OPTIONS.map((hour) => (
                    <option key={`start-hour-${hour}`} value={hour}>
                      {hour}
                    </option>
                  ))}
                </select>
                <span className="text-lg font-semibold text-[color:var(--ink-muted)]">:</span>
                <select
                  value={startTimeParts.minute}
                  onChange={(event) =>
                    setFormState((current) => ({
                      ...current,
                      startTime: replaceTimePart(current.startTime, "minute", event.target.value),
                    }))
                  }
                  className="w-full rounded-[1.25rem] border border-[color:var(--stroke-strong)] bg-white px-4 py-3 text-[color:var(--ink-strong)] outline-none focus:border-[color:var(--accent)]"
                  aria-label="Startminute"
                  required
                >
                  {MINUTE_OPTIONS.map((minute) => (
                    <option key={`start-minute-${minute}`} value={minute}>
                      {minute}
                    </option>
                  ))}
                </select>
              </div>
              {minimumTimeForSelectedStartDate && (
                <p className="mt-2 text-xs text-[color:var(--ink-muted)]">
                  Heute frühestens ab {minimumTimeForSelectedStartDate} Uhr buchbar.
                </p>
              )}
            </label>

            <label className="block">
              <span className="text-xs font-semibold uppercase tracking-[0.16em] text-[color:var(--ink-muted)]">
                Enddatum
              </span>
              <input
                type="date"
                value={formState.endDate}
                min={formState.startDate}
                onChange={(event) => setFormState((current) => ({ ...current, endDate: event.target.value }))}
                className="mt-2 w-full rounded-[1.25rem] border border-[color:var(--stroke-strong)] bg-white px-4 py-3 text-[color:var(--ink-strong)] outline-none focus:border-[color:var(--accent)]"
                required
              />
            </label>

            <label className="block">
              <span className="text-xs font-semibold uppercase tracking-[0.16em] text-[color:var(--ink-muted)]">
                Endzeit
              </span>
              <div className="mt-2 grid grid-cols-[1fr_auto_1fr] items-center gap-3">
                <select
                  value={endTimeParts.hour}
                  onChange={(event) =>
                    setFormState((current) => ({
                      ...current,
                      endTime: replaceTimePart(current.endTime, "hour", event.target.value),
                    }))
                  }
                  className="w-full rounded-[1.25rem] border border-[color:var(--stroke-strong)] bg-white px-4 py-3 text-[color:var(--ink-strong)] outline-none focus:border-[color:var(--accent)]"
                  aria-label="Endstunde"
                  required
                >
                  {HOUR_OPTIONS.map((hour) => (
                    <option key={`end-hour-${hour}`} value={hour}>
                      {hour}
                    </option>
                  ))}
                </select>
                <span className="text-lg font-semibold text-[color:var(--ink-muted)]">:</span>
                <select
                  value={endTimeParts.minute}
                  onChange={(event) =>
                    setFormState((current) => ({
                      ...current,
                      endTime: replaceTimePart(current.endTime, "minute", event.target.value),
                    }))
                  }
                  className="w-full rounded-[1.25rem] border border-[color:var(--stroke-strong)] bg-white px-4 py-3 text-[color:var(--ink-strong)] outline-none focus:border-[color:var(--accent)]"
                  aria-label="Endminute"
                  required
                >
                  {MINUTE_OPTIONS.map((minute) => (
                    <option key={`end-minute-${minute}`} value={minute}>
                      {minute}
                    </option>
                  ))}
                </select>
              </div>
              {minimumTimeForSelectedEndDate && (
                <p className="mt-2 text-xs text-[color:var(--ink-muted)]">
                  Am selben Tag muss die Endzeit nach {minimumTimeForSelectedEndDate} Uhr liegen.
                </p>
              )}
            </label>
          </div>

          <div className="rounded-[1.5rem] bg-[color:var(--surface-muted)] px-4 py-4 text-sm leading-7 text-[color:var(--ink-soft)]">
            <p>Regel im MVP: 1 Buchung = 1 geplanter Bibliotheksplatz.</p>
            <p>Datum und Uhrzeit sind frei wählbar, solange der Start in der Zukunft liegt.</p>
            <p>Buchungen sind nur innerhalb eines einzelnen Öffnungstags erlaubt.</p>
            <p>Die Kurzfristprognose übernimmt das Overlay automatisch für bestätigte Zeitfenster.</p>
          </div>

          {submitError && (
            <div className="rounded-[1.5rem] border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
              {submitError}
            </div>
          )}
          {submitSuccess && (
            <div className="rounded-[1.5rem] border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-700">
              {submitSuccess}
            </div>
          )}
          {bookingsError && (
            <div className="rounded-[1.5rem] border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-700">
              {bookingsError}
            </div>
          )}

          <button
            type="submit"
            disabled={pending}
            className="inline-flex items-center gap-2 rounded-full bg-[color:var(--accent)] px-5 py-3 text-sm font-semibold text-white transition hover:bg-[color:var(--accent-strong)] disabled:cursor-not-allowed disabled:opacity-60"
          >
            {pending ? "Speichere …" : "Buchung speichern"}
          </button>
        </form>
      </section>

      <section className="grid gap-6">
        <article className="rounded-[2rem] border border-[color:var(--stroke-soft)] bg-[color:var(--surface-raised)] p-6 shadow-[0_24px_54px_rgba(15,23,42,0.08)]">
          <div className="flex items-end justify-between gap-4">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[color:var(--ink-muted)]">
                Aktive Buchungen
              </p>
              <h3 className="mt-1 text-2xl font-semibold text-[color:var(--ink-strong)]">
                Kommende Zeitfenster
              </h3>
            </div>
            <p className="text-sm text-[color:var(--ink-soft)]">
              {bookingsLoading ? "Lade …" : `${upcomingBookings.length} aktiv`}
            </p>
          </div>

          <div className="mt-5 grid gap-3">
            {upcomingBookings.length > 0 ? (
              upcomingBookings.map((booking) => (
                <div
                  key={booking.booking_id}
                  className="rounded-[1.5rem] border border-[color:var(--stroke-soft)] bg-white px-4 py-4"
                >
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                    <div className="text-sm leading-7 text-[color:var(--ink-soft)]">
                      <p className="font-semibold text-[color:var(--ink-strong)]">
                        {formatDateTime(booking.starts_at)} bis {formatDateTime(booking.ends_at)}
                      </p>
                      <p>Status: {booking.status}</p>
                    </div>
                    <button
                      type="button"
                      onClick={() => handleCancel(booking.booking_id)}
                      disabled={cancellingId === booking.booking_id}
                      className="rounded-full border border-rose-200 px-4 py-2 text-sm font-semibold text-rose-700 transition hover:border-rose-300 hover:bg-rose-50 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {cancellingId === booking.booking_id ? "Storniere …" : "Stornieren"}
                    </button>
                  </div>
                </div>
              ))
            ) : (
              <div className="rounded-[1.5rem] border border-dashed border-[color:var(--stroke-strong)] bg-[color:var(--surface-muted)] px-4 py-4 text-sm text-[color:var(--ink-soft)]">
                Noch keine bestätigten zukünftigen Buchungen vorhanden.
              </div>
            )}
          </div>
        </article>

        <article className="rounded-[2rem] border border-[color:var(--stroke-soft)] bg-[color:var(--surface-raised)] p-6 shadow-[0_24px_54px_rgba(15,23,42,0.08)]">
          <div className="flex items-end justify-between gap-4">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[color:var(--ink-muted)]">
                Verlauf
              </p>
              <h3 className="mt-1 text-2xl font-semibold text-[color:var(--ink-strong)]">
                Historie und stornierte Slots
              </h3>
            </div>
            <p className="text-sm text-[color:var(--ink-soft)]">{historyBookings.length} Einträge</p>
          </div>

          <div className="mt-5 grid gap-3">
            {historyBookings.length > 0 ? (
              historyBookings.map((booking) => (
                <div
                  key={booking.booking_id}
                  className="rounded-[1.5rem] border border-[color:var(--stroke-soft)] bg-white px-4 py-4 text-sm leading-7 text-[color:var(--ink-soft)]"
                >
                  <p className="font-semibold text-[color:var(--ink-strong)]">
                    {formatDateTime(booking.starts_at)} bis {formatDateTime(booking.ends_at)}
                  </p>
                  <p>Status: {booking.status}</p>
                  {booking.cancelled_at && <p>Storniert am: {formatDateTime(booking.cancelled_at)}</p>}
                </div>
              ))
            ) : (
              <div className="rounded-[1.5rem] border border-dashed border-[color:var(--stroke-strong)] bg-[color:var(--surface-muted)] px-4 py-4 text-sm text-[color:var(--ink-soft)]">
                Noch keine historischen oder stornierten Buchungen vorhanden.
              </div>
            )}
          </div>
        </article>
      </section>
    </div>
  );
}
