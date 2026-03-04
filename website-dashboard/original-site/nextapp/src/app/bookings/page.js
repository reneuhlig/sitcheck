"use client";

import { useMemo } from "react";
import Link from "next/link";
import { useAppData } from "@/context/AppDataContext";

// Überschriftenformat für Tagesabschnitte.
function formatDateHeading(isoString) {
  const date = new Date(isoString);
  return date.toLocaleDateString("de-DE", {
    weekday: "long",
    day: "numeric",
    month: "long",
  });
}

// Formatiert Start- und Endzeit im Stil „10:00 – 11:00 Uhr“.
function formatTimeRange(startIso, endIso) {
  const start = new Date(startIso);
  const end = new Date(endIso);
  const timeOptions = { hour: "2-digit", minute: "2-digit" };
  return `${start.toLocaleTimeString("de-DE", timeOptions)} – ${end.toLocaleTimeString("de-DE", timeOptions)} Uhr`;
}

export default function BookingsPage() {
  const { bookings, cancelBooking } = useAppData();

  const { upcoming, past } = useMemo(() => {
    // Trenne aktuelle und vergangene Reservierungen und halte die Liste chronologisch.
    const now = new Date();
    const sorted = [...bookings].sort(
      (a, b) => new Date(a.startDateTime).getTime() - new Date(b.startDateTime).getTime(),
    );
    return {
      upcoming: sorted.filter((booking) => new Date(booking.endDateTime) >= now),
      past: sorted.filter((booking) => new Date(booking.endDateTime) < now),
    };
  }, [bookings]);

  const groupedUpcoming = useMemo(() => {
    // Gruppiert aktive Termine nach Tag, damit die Darstellung kompakt bleibt.
    return upcoming.reduce((acc, booking) => {
      const key = new Date(booking.startDateTime).toDateString();
      if (!acc[key]) acc[key] = [];
      acc[key].push(booking);
      return acc;
    }, {});
  }, [upcoming]);

  const hasBookings = upcoming.length > 0 || past.length > 0;

  return (
    <div className="mx-auto flex w-full max-w-4xl flex-col gap-8 px-4 pb-10 pt-6 sm:px-6 sm:pt-8">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900 sm:text-3xl">Buchungen verwalten</h1>
          <p className="text-sm text-slate-600">
            Behalte alle Reservierungen im Blick, storniere bei Bedarf und plane neue Anfragen.
          </p>
        </div>
        <Link
          href="/bookings/new"
          className="inline-flex items-center gap-2 rounded-full bg-sky-500 px-4 py-2 font-semibold text-white shadow-sm transition hover:bg-sky-600"
        >
          <span aria-hidden="true">＋</span>
          Neue Buchung starten
        </Link>
      </div>

      {!hasBookings && (
        <div className="rounded-3xl border border-dashed border-sky-200 bg-sky-50/60 p-8 text-center text-slate-600">
          <h2 className="text-xl font-semibold text-slate-900">Noch keine Reservierungen</h2>
          <p className="mt-2 text-sm">
            Starte über den Button oben eine unverbindliche Buchung oder markiere Räume als Favorit, um sie schneller wiederzufinden.
          </p>
        </div>
      )}

      {Object.entries(groupedUpcoming).map(([dateKey, dayBookings]) => (
        <section key={dateKey} className="space-y-4 rounded-3xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
          <div className="flex items-baseline justify-between">
            <div>
              <h2 className="text-lg font-semibold text-slate-900">{formatDateHeading(dayBookings[0].startDateTime)}</h2>
              <p className="text-sm text-slate-500">Aktive Reservierungen für diesen Tag.</p>
            </div>
            <span className="text-xs font-medium uppercase tracking-wide text-slate-400">
              {dayBookings.length} Buchung{dayBookings.length === 1 ? "" : "en"}
            </span>
          </div>
          <div className="grid gap-3">
            {dayBookings.map((booking) => (
              <article
                key={booking.id}
                className="flex flex-col gap-4 rounded-2xl border border-slate-200 bg-slate-50/70 p-4 text-sm text-slate-600 shadow-inner sm:flex-row sm:items-center sm:justify-between"
              >
                <div className="space-y-1">
                  <p className="text-sm font-semibold text-slate-900">{booking.roomName}</p>
                  <p className="text-xs uppercase tracking-wide text-slate-500">{booking.category}</p>
                  <p>{formatTimeRange(booking.startDateTime, booking.endDateTime)}</p>
                  <span className="inline-flex w-fit items-center rounded-full bg-sky-100 px-3 py-1 text-xs font-semibold text-sky-700">
                    {booking.status}
                  </span>
                </div>
                <div className="flex flex-col items-start gap-2 sm:items-end">
                  <Link
                    href={`/rooms/${booking.roomId}`}
                    className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold text-slate-600 transition hover:border-sky-200 hover:text-sky-700"
                  >
                    Raumdetails
                    <span aria-hidden="true">→</span>
                  </Link>
                  <button
                    type="button"
                    onClick={() => cancelBooking(booking.id)}
                    className="inline-flex items-center gap-2 rounded-full border border-rose-200 bg-rose-50 px-3 py-1 text-xs font-semibold text-rose-600 transition hover:border-rose-300 hover:text-rose-700"
                  >
                    Buchung stornieren
                  </button>
                </div>
              </article>
            ))}
          </div>
        </section>
      ))}

      {past.length > 0 && (
        <section className="space-y-4 rounded-3xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
          <div className="flex items-baseline justify-between">
            <div>
              <h2 className="text-lg font-semibold text-slate-900">Vergangene Buchungen</h2>
              <p className="text-sm text-slate-500">Archiv deiner abgeschlossenen Reservierungen.</p>
            </div>
            <span className="text-xs font-medium uppercase tracking-wide text-slate-400">
              {past.length} Buchung{past.length === 1 ? "" : "en"}
            </span>
          </div>
          <div className="grid gap-3 text-sm text-slate-600">
            {past.map((booking) => (
              <div
                key={`past-${booking.id}`}
                className="flex flex-col gap-2 rounded-2xl border border-slate-200 bg-slate-50/60 p-4 sm:flex-row sm:items-center sm:justify-between"
              >
                <div>
                  <p className="text-sm font-semibold text-slate-900">{booking.roomName}</p>
                  <p className="text-xs uppercase tracking-wide text-slate-500">{booking.category}</p>
                  <p>{formatTimeRange(booking.startDateTime, booking.endDateTime)}</p>
                </div>
                <span className="inline-flex items-center rounded-full bg-slate-200 px-3 py-1 text-xs font-semibold text-slate-600">
                  Abgeschlossen
                </span>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
