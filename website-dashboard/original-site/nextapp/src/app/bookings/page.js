"use client";

import { useState } from "react";
import Link from "next/link";
import { useAppData } from "@/context/AppDataContext";

function toLocalDateTimeValue(date) {
  const localDate = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
  return localDate.toISOString().slice(0, 16);
}

function defaultStartValue() {
  const date = new Date();
  date.setMinutes(date.getMinutes() + 30);
  date.setSeconds(0, 0);
  return toLocalDateTimeValue(date);
}

function defaultEndValue() {
  const date = new Date();
  date.setMinutes(date.getMinutes() + 90);
  date.setSeconds(0, 0);
  return toLocalDateTimeValue(date);
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

  const [formState, setFormState] = useState({
    startsAt: defaultStartValue(),
    endsAt: defaultEndValue(),
  });
  const [submitError, setSubmitError] = useState("");
  const [submitSuccess, setSubmitSuccess] = useState("");
  const [pending, setPending] = useState(false);
  const [cancellingId, setCancellingId] = useState("");

  const now = new Date();
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
      await createBooking({
        starts_at: new Date(formState.startsAt).toISOString(),
        ends_at: new Date(formState.endsAt).toISOString(),
      });
      setSubmitSuccess("Buchung gespeichert. Das Zeitfenster fließt jetzt in die Kurzfristprognose ein.");
      setFormState({
        startsAt: defaultStartValue(),
        endsAt: defaultEndValue(),
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Buchung konnte nicht erstellt werden.";
      setSubmitError(message);
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

        <form onSubmit={handleSubmit} className="mt-6 space-y-5">
          <label className="block">
            <span className="text-xs font-semibold uppercase tracking-[0.16em] text-[color:var(--ink-muted)]">
              Start
            </span>
            <input
              type="datetime-local"
              value={formState.startsAt}
              min={toLocalDateTimeValue(new Date())}
              onChange={(event) => setFormState((current) => ({ ...current, startsAt: event.target.value }))}
              className="mt-2 w-full rounded-[1.25rem] border border-[color:var(--stroke-strong)] bg-white px-4 py-3 text-[color:var(--ink-strong)] outline-none focus:border-[color:var(--accent)]"
              required
            />
          </label>

          <label className="block">
            <span className="text-xs font-semibold uppercase tracking-[0.16em] text-[color:var(--ink-muted)]">
              Ende
            </span>
            <input
              type="datetime-local"
              value={formState.endsAt}
              min={formState.startsAt}
              onChange={(event) => setFormState((current) => ({ ...current, endsAt: event.target.value }))}
              className="mt-2 w-full rounded-[1.25rem] border border-[color:var(--stroke-strong)] bg-white px-4 py-3 text-[color:var(--ink-strong)] outline-none focus:border-[color:var(--accent)]"
              required
            />
          </label>

          <div className="rounded-[1.5rem] bg-[color:var(--surface-muted)] px-4 py-4 text-sm leading-7 text-[color:var(--ink-soft)]">
            <p>Regel im MVP: 1 Buchung = 1 geplanter Bibliotheksplatz.</p>
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
