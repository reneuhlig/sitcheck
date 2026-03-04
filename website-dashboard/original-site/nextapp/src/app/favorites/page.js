"use client";

import Link from "next/link";
import { useMemo } from "react";
import { rooms } from "@/data/rooms";
import { useAppData } from "@/context/AppDataContext";

// Farbliche Hervorhebung für Favoriten je Kategorie.
const CATEGORY_COLORS = {
  "Lesesäle": "from-sky-400 via-sky-500 to-sky-600",
  Gruppenräume: "from-amber-400 via-orange-400 to-orange-500",
  Seitenbänke: "from-emerald-400 via-teal-400 to-teal-500",
};

export default function FavoritesPage() {
  const { favorites, toggleFavorite } = useAppData();

  const favoriteRooms = useMemo(
    // Filtert alle Räume anhand der gemerkten IDs.
    () => rooms.filter((room) => favorites.includes(room.id)),
    [favorites],
  );

  const groupedFavorites = useMemo(() => {
    // Gruppiert Favoriten nach Bereich für eine geordnete Ausgabe.
    return favoriteRooms.reduce((acc, room) => {
      if (!acc[room.type]) acc[room.type] = [];
      acc[room.type].push(room);
      return acc;
    }, {});
  }, [favoriteRooms]);

  const hasFavorites = favoriteRooms.length > 0;

  return (
    <div className="mx-auto flex w-full max-w-4xl flex-col gap-8 px-4 pb-10 pt-6 sm:px-6 sm:pt-8">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900 sm:text-3xl">Meine Favoriten</h1>
          <p className="text-sm text-slate-600">
            Merke dir bevorzugte Räume, um sie schneller zu finden und direkt zu buchen.
          </p>
        </div>
        <Link
          href="/bookings/new"
          className="inline-flex items-center gap-2 rounded-full border border-sky-200 bg-white px-4 py-2 text-sm font-semibold text-sky-700 shadow-sm transition hover:border-sky-300 hover:text-sky-800"
        >
          Lieblingsraum buchen
        </Link>
      </div>

      {!hasFavorites && (
        <div className="rounded-3xl border border-dashed border-slate-200 bg-slate-50/70 p-8 text-center text-slate-600">
          <h2 className="text-xl font-semibold text-slate-900">Noch keine Favoriten gespeichert</h2>
          <p className="mt-2 text-sm">
            Tippe auf „Merken“ in der Raumübersicht oder der Detailseite, um Räume hier abzulegen.
          </p>
          <Link
            href="/"
            className="mt-4 inline-flex items-center gap-2 rounded-full bg-sky-500 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-sky-600"
          >
            Zur Raumübersicht
          </Link>
        </div>
      )}

      {Object.entries(groupedFavorites).map(([type, typeRooms]) => (
        <section key={type} className="space-y-4 rounded-3xl border border-slate-200 bg-white p-6 shadow-sm sm:p-7">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span
                className={`inline-flex items-center gap-2 rounded-full bg-gradient-to-r ${
                  CATEGORY_COLORS[type] ?? "from-slate-400 via-slate-500 to-slate-600"
                } px-4 py-1 text-xs font-semibold uppercase tracking-wide text-white shadow`}
              >
                {type}
              </span>
              <span className="text-xs font-medium uppercase tracking-wide text-slate-500">
                {typeRooms.length} Favorit{typeRooms.length === 1 ? "" : "en"}
              </span>
            </div>
            <Link
              href="/"
              className="text-xs font-semibold text-sky-600 transition hover:text-sky-700"
            >
              Weitere Räume entdecken →
            </Link>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            {typeRooms.map((room) => {
              const utilization = Math.round((room.people / room.capacity) * 100);
              return (
                <article
                  key={room.id}
                  className="flex flex-col gap-4 rounded-3xl border border-slate-200 bg-gradient-to-br from-white via-white to-slate-50 p-5 shadow-sm"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="space-y-1">
                      <p className="text-xs uppercase tracking-wide text-slate-400">{room.type}</p>
                      <h2 className="text-lg font-semibold text-slate-900">{room.name}</h2>
                    </div>
                    <button
                      type="button"
                      onClick={() => toggleFavorite(room.id)}
                      className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold text-slate-600 transition hover:border-rose-300 hover:text-rose-500"
                    >
                      Entfernen
                    </button>
                  </div>
                  <div className="grid grid-cols-2 gap-3 text-xs text-slate-500">
                    <div>
                      <p>Aktuelle Personen</p>
                      <p className="text-sm font-semibold text-slate-900">{room.people}</p>
                    </div>
                    <div>
                      <p>Kapazität</p>
                      <p className="text-sm font-semibold text-slate-900">{room.capacity}</p>
                    </div>
                    <div>
                      <p>Auslastung</p>
                      <p className="text-sm font-semibold text-slate-900">{utilization}%</p>
                    </div>
                    <div>
                      <p>Freie Plätze</p>
                      <p className="text-sm font-semibold text-slate-900">{Math.max(room.capacity - room.people, 0)}</p>
                    </div>
                  </div>
                  <div className="flex flex-wrap items-center justify-between gap-2 text-xs font-semibold text-sky-600">
                    <Link
                      href={`/rooms/${room.id}`}
                      className="inline-flex items-center gap-2 transition hover:text-sky-700"
                    >
                      Details ansehen
                      <span aria-hidden="true">→</span>
                    </Link>
                    <span className="rounded-full bg-slate-200 px-3 py-1 text-slate-600">
                      Letzter Stand: {utilization}%
                    </span>
                  </div>
                </article>
              );
            })}
          </div>
        </section>
      ))}
    </div>
  );
}
