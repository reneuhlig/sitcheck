"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { rooms } from "@/data/rooms";
import { useAppData } from "@/context/AppDataContext";

// Auswahlmöglichkeiten für häufig genutzte Buchungsdauern.
const durationOptions = [
  { label: "30 Minuten", value: 30 },
  { label: "60 Minuten", value: 60 },
  { label: "90 Minuten", value: 90 },
  { label: "120 Minuten", value: 120 },
];

export default function NewBookingPage() {
  const { bookings, setBookings } = useAppData();
  const router = useRouter();
  const today = useMemo(() => new Date().toISOString().split("T")[0], []);
  const [formData, setFormData] = useState({
    roomId: rooms[0]?.id ?? 1,
    date: today,
    startTime: "12:00",
    duration: 60,
    status: "Anfrage",
    notes: "",
  });
  const [confirmation, setConfirmation] = useState(null);

  // Räume nach Kategorie gruppieren, damit die Dropdown-Liste strukturiert bleibt.
  const groupedRooms = useMemo(() => {
    return rooms.reduce((acc, room) => {
      if (!acc[room.type]) acc[room.type] = [];
      acc[room.type].push(room);
      return acc;
    }, {});
  }, []);

  const handleChange = (field) => (event) => {
    const value = field === "duration" ? Number(event.target.value) : event.target.value;
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    const room = rooms.find((entry) => entry.id === Number(formData.roomId));
    if (!room) return;

    const [startHour, startMinute] = formData.startTime.split(":").map(Number);
    const start = new Date(formData.date);
    start.setHours(startHour, startMinute, 0, 0);
    const end = new Date(start.getTime() + formData.duration * 60 * 1000);

    const newId = bookings.reduce((max, booking) => Math.max(max, booking.id), 0) + 1;
    const newBooking = {
      id: newId,
      roomId: room.id,
      roomName: room.name,
      category: room.type,
      startDateTime: start.toISOString(),
      endDateTime: end.toISOString(),
      status: formData.status,
      notes: formData.notes,
    };

    setBookings([...bookings, newBooking]);
    setConfirmation(newBooking);
  };

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-8 px-4 pb-10 pt-6 sm:px-6 sm:pt-8">
      <div className="space-y-3">
        <h1 className="text-2xl font-semibold text-slate-900 sm:text-3xl">Unverbindliche Buchung anlegen</h1>
        <p className="text-sm text-slate-600">
          Reserviere dir einen Raum im Lesesaal für einen gewünschten Zeitraum. Die Anfrage wird lokal gespeichert und kann später an das Backend übergeben werden.
        </p>
        <div className="flex flex-wrap gap-2 text-xs text-slate-500">
          <span className="inline-flex items-center rounded-full bg-sky-100 px-3 py-1 font-semibold text-sky-700">
            Schritt 1 · Anfrage erstellen
          </span>
          <span className="inline-flex items-center rounded-full bg-slate-200 px-3 py-1 font-semibold text-slate-700">
            Schritt 2 · Bestätigung folgt
          </span>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6 rounded-3xl border border-slate-200 bg-white p-6 shadow-sm sm:p-7">
        <div className="grid gap-4 sm:grid-cols-2">
          <label className="flex flex-col gap-2 text-sm text-slate-600">
            <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">Kategorie & Raum</span>
            <select
              value={formData.roomId}
              onChange={handleChange("roomId")}
              className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-medium text-slate-700 shadow-inner focus:border-sky-400 focus:outline-none focus:ring-2 focus:ring-sky-200"
            >
              {Object.entries(groupedRooms).map(([type, typeRooms]) => (
                <optgroup key={type} label={type}>
                  {typeRooms.map((room) => (
                    <option key={room.id} value={room.id}>
                      {room.name} · {room.capacity} Plätze
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-2 text-sm text-slate-600">
            <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">Datum</span>
            <input
              type="date"
              min={today}
              value={formData.date}
              onChange={handleChange("date")}
              className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-medium text-slate-700 shadow-inner focus:border-sky-400 focus:outline-none focus:ring-2 focus:ring-sky-200"
              required
            />
          </label>
          <label className="flex flex-col gap-2 text-sm text-slate-600">
            <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">Startzeit</span>
            <input
              type="time"
              value={formData.startTime}
              onChange={handleChange("startTime")}
              className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-medium text-slate-700 shadow-inner focus:border-sky-400 focus:outline-none focus:ring-2 focus:ring-sky-200"
              required
            />
          </label>
          <label className="flex flex-col gap-2 text-sm text-slate-600">
            <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">Dauer</span>
            <select
              value={formData.duration}
              onChange={handleChange("duration")}
              className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-medium text-slate-700 shadow-inner focus:border-sky-400 focus:outline-none focus:ring-2 focus:ring-sky-200"
            >
              {durationOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="grid gap-4 sm:grid-cols-2">
          <label className="flex flex-col gap-2 text-sm text-slate-600">
            <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">Status</span>
            <select
              value={formData.status}
              onChange={handleChange("status")}
              className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-medium text-slate-700 shadow-inner focus:border-sky-400 focus:outline-none focus:ring-2 focus:ring-sky-200"
            >
              <option value="Anfrage">Anfrage</option>
              <option value="Option">Option</option>
              <option value="Bestätigt">Bestätigt</option>
            </select>
          </label>
          <label className="flex flex-col gap-2 text-sm text-slate-600">
            <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">Notiz</span>
            <textarea
              value={formData.notes}
              onChange={handleChange("notes")}
              rows={3}
              placeholder="Welche Gruppe nutzt den Raum? Benötigte Ausstattung?"
              className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-medium text-slate-700 shadow-inner focus:border-sky-400 focus:outline-none focus:ring-2 focus:ring-sky-200"
            />
          </label>
        </div>
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <p className="text-xs text-slate-500">
            Deine Anfrage wird lokal gespeichert. Du kannst sie später über „Buchungen verwalten“ bearbeiten oder stornieren.
          </p>
          <button
            type="submit"
            className="inline-flex items-center gap-2 rounded-full bg-sky-500 px-5 py-2 font-semibold text-white shadow-sm transition hover:bg-sky-600"
          >
            Anfrage speichern
          </button>
        </div>
      </form>

      {confirmation && (
        <div className="space-y-3 rounded-3xl border border-emerald-200 bg-emerald-50 p-6 text-sm text-emerald-700">
          <div className="flex items-center justify-between">
            <p className="text-base font-semibold text-emerald-800">Anfrage gespeichert!</p>
            <button
              type="button"
              onClick={() => {
                setConfirmation(null);
                router.push("/bookings");
              }}
              className="inline-flex items-center gap-2 rounded-full border border-emerald-300 bg-white/80 px-3 py-1 text-xs font-semibold text-emerald-700 transition hover:border-emerald-400"
            >
              Zu meinen Buchungen
            </button>
          </div>
          <p>
            {confirmation.roomName} · {new Date(confirmation.startDateTime).toLocaleDateString("de-DE", {
              weekday: "long",
              day: "numeric",
              month: "long",
            })}
          </p>
          <p>
            Zeitraum: {new Date(confirmation.startDateTime).toLocaleTimeString("de-DE", {
              hour: "2-digit",
              minute: "2-digit",
            })}
            {" "}–{" "}
            {new Date(confirmation.endDateTime).toLocaleTimeString("de-DE", {
              hour: "2-digit",
              minute: "2-digit",
            })}
            {" "}Uhr
          </p>
        </div>
      )}
    </div>
  );
}
