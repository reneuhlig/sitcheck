"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useAppData } from "@/context/AppDataContext";
import { sitcheckApi } from "@/lib/api";

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

function formatDayLabel(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Unbekannt";

  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const tomorrow = new Date(today);
  tomorrow.setDate(tomorrow.getDate() + 1);

  const bookingDay = new Date(date);
  bookingDay.setHours(0, 0, 0, 0);

  if (bookingDay.getTime() === today.getTime()) return "Heute";
  if (bookingDay.getTime() === tomorrow.getTime()) return "Morgen";
  return date.toLocaleDateString("de-DE", {
    weekday: "short",
    day: "2-digit",
    month: "2-digit",
  });
}

export default function AdminBookingsPage() {
  const { authenticated, sessionLoading, user } = useAppData();
  const [bookings, setBookings] = useState([]);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [sortOrder, setSortOrder] = useState("oldest");

  useEffect(() => {
    const abortController = new AbortController();
    async function syncBookings() {
      if (!authenticated || user?.role !== "admin") {
        setBookings([]);
        setErrorMessage("");
        return;
      }

      setLoading(true);
      try {
        const payload = await sitcheckApi.getAdminBookings(abortController.signal);
        setBookings(Array.isArray(payload) ? payload : []);
        setErrorMessage("");
      } catch (error) {
        if (abortController.signal.aborted || error?.name === "AbortError") {
          return;
        }
        const message =
          error instanceof Error ? error.message : "Buchungsübersicht konnte nicht geladen werden.";
        setErrorMessage(message);
      } finally {
        setLoading(false);
      }
    }

    syncBookings();
    return () => abortController.abort();
  }, [authenticated, user?.role]);

  async function handleReload() {
    if (!authenticated || user?.role !== "admin") {
      setBookings([]);
      setErrorMessage("");
      return;
    }

    setLoading(true);
    try {
      const payload = await sitcheckApi.getAdminBookings();
      setBookings(Array.isArray(payload) ? payload : []);
      setErrorMessage("");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Buchungsübersicht konnte nicht geladen werden.";
      setErrorMessage(message);
    } finally {
      setLoading(false);
    }
  }

  const sortedBookings = [...bookings].sort((left, right) => {
    const leftTs = new Date(left.starts_at).getTime();
    const rightTs = new Date(right.starts_at).getTime();
    return sortOrder === "newest" ? rightTs - leftTs : leftTs - rightTs;
  });

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
          Globale Buchungen sind nur nach Login sichtbar
        </h2>
        <p className="mt-4 max-w-2xl text-sm leading-7 text-[color:var(--ink-soft)]">
          Diese Ansicht zeigt Buchungen aller Nutzer ab dem heutigen Tag. Dafür ist zunächst eine
          Anmeldung erforderlich.
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

  if (user?.role !== "admin") {
    return (
      <section className="rounded-[2rem] border border-[color:var(--stroke-soft)] bg-[color:var(--surface-raised)] p-8 shadow-[0_24px_54px_rgba(15,23,42,0.08)]">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[color:var(--ink-muted)]">
          Admin-Bereich
        </p>
        <h2 className="mt-2 text-3xl font-semibold text-[color:var(--ink-strong)]">
          Für diese Ansicht werden Admin-Rechte benötigt
        </h2>
        <p className="mt-4 max-w-2xl text-sm leading-7 text-[color:var(--ink-soft)]">
          Die globale Buchungsübersicht enthält Buchungen aller Nutzer und ist deshalb nur für Admins
          freigeschaltet.
        </p>
        <Link
          href="/bookings"
          className="mt-6 inline-flex items-center gap-2 rounded-full bg-[color:var(--accent)] px-4 py-2 text-sm font-semibold text-white transition hover:bg-[color:var(--accent-strong)]"
        >
          Zu meinen Buchungen
          <span aria-hidden="true">→</span>
        </Link>
      </section>
    );
  }

  return (
    <div className="grid gap-6">
      <section className="rounded-[2rem] border border-[color:var(--stroke-soft)] bg-[color:var(--surface-raised)] p-6 shadow-[0_24px_54px_rgba(15,23,42,0.08)]">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[color:var(--ink-muted)]">
              Admin-Übersicht
            </p>
            <h2 className="mt-2 text-3xl font-semibold text-[color:var(--ink-strong)]">
              Buchungen aller Nutzer
            </h2>
            <p className="mt-3 max-w-3xl text-sm leading-7 text-[color:var(--ink-soft)]">
              Angezeigt werden alle bestätigten Buchungen, die noch in den heutigen Tag fallen oder an
              einem späteren Datum liegen.
            </p>
          </div>

          <button
            type="button"
            onClick={handleReload}
            disabled={loading}
            className="inline-flex items-center justify-center gap-2 rounded-full border border-[color:var(--stroke-strong)] bg-white px-4 py-2 text-sm font-semibold text-[color:var(--ink-strong)] transition hover:border-[color:var(--accent-soft)] hover:text-[color:var(--accent-strong)] disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading ? "Lade …" : "Neu laden"}
          </button>
        </div>
      </section>

      {errorMessage && (
        <div className="rounded-[1.5rem] border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
          {errorMessage}
        </div>
      )}

      <section className="rounded-[2rem] border border-[color:var(--stroke-soft)] bg-[color:var(--surface-raised)] p-6 shadow-[0_24px_54px_rgba(15,23,42,0.08)]">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[color:var(--ink-muted)]">
              Nutzerbuchungen
            </p>
            <h3 className="mt-1 text-2xl font-semibold text-[color:var(--ink-strong)]">
              Bestätigte Zeitfenster
            </h3>
          </div>
          <div className="flex flex-col gap-3 lg:items-end">
            <div className="flex flex-wrap gap-2">
              {[
                { id: "oldest", label: "Älteste zuerst" },
                { id: "newest", label: "Neueste zuerst" },
              ].map((item) => {
                const active = sortOrder === item.id;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => setSortOrder(item.id)}
                    className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
                      active
                        ? "bg-[color:var(--accent)] text-white shadow-[0_12px_28px_rgba(139,198,236,0.38)]"
                        : "border border-[color:var(--stroke-strong)] bg-white text-[color:var(--ink-strong)] hover:border-[color:var(--accent-soft)] hover:text-[color:var(--accent-strong)]"
                    }`}
                  >
                    {item.label}
                  </button>
                );
              })}
            </div>
            <p className="text-sm text-[color:var(--ink-soft)]">{loading ? "Aktualisiere …" : `${bookings.length} Einträge`}</p>
          </div>
        </div>

        <div className="mt-5 grid gap-3">
          {sortedBookings.length > 0 ? (
            sortedBookings.map((booking) => (
              <article
                key={booking.booking_id}
                className="rounded-[1.5rem] border border-[color:var(--stroke-soft)] bg-white px-4 py-4"
              >
                <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[color:var(--ink-muted)]">
                      Gebucht von
                    </p>
                    <h4 className="mt-1 text-lg font-semibold text-[color:var(--ink-strong)]">
                      {booking.username}
                    </h4>
                    <p className="mt-2 text-sm leading-7 text-[color:var(--ink-soft)]">
                      {formatDateTime(booking.starts_at)} bis {formatDateTime(booking.ends_at)}
                    </p>
                  </div>

                  <div className="rounded-[1.25rem] bg-[color:var(--surface-muted)] px-4 py-3 text-sm text-[color:var(--ink-soft)]">
                    <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[color:var(--ink-muted)]">
                      Zeitraum
                    </p>
                    <p className="mt-1 font-semibold text-[color:var(--ink-strong)]">{formatDayLabel(booking.starts_at)}</p>
                    <p className="mt-1 text-xs">Erstellt: {formatDateTime(booking.created_at)}</p>
                  </div>
                </div>
              </article>
            ))
          ) : (
            <div className="rounded-[1.5rem] border border-dashed border-[color:var(--stroke-strong)] bg-[color:var(--surface-muted)] px-4 py-4 text-sm text-[color:var(--ink-soft)]">
              Für heute und die kommenden Tage sind derzeit keine bestätigten Buchungen vorhanden.
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
