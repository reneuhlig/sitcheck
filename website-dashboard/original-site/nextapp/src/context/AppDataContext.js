"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
} from "react";

export const AppDataContext = createContext(null);

const INITIAL_FAVORITES = [7, 3, 11];

// Erstellt dynamische Buchungsdaten, damit Demo-Listen immer aktuelle Zeitstempel haben.
const createInitialBookings = () => {
  const templates = [
    {
      id: 1,
      roomId: 7,
      roomName: "Gruppenraum B",
      category: "Gruppenräume",
      dayOffset: 0,
      hour: 13,
      minute: 30,
      durationMinutes: 90,
      status: "Bestätigt",
      notes: "Projektbesprechung mit dem Tutorium.",
    },
    {
      id: 2,
      roomId: 3,
      roomName: "Lesesaal 3",
      category: "Lesesäle",
      dayOffset: 1,
      hour: 9,
      minute: 0,
      durationMinutes: 120,
      status: "Option",
      notes: "Prüfungsvorbereitung Informatik.",
    },
    {
      id: 3,
      roomId: 11,
      roomName: "Seitenbank 1",
      category: "Seitenbänke",
      dayOffset: -1,
      hour: 18,
      minute: 0,
      durationMinutes: 60,
      status: "Abgeschlossen",
      notes: "Konzentrationsphase vor dem Vortrag.",
    },
  ];

  return templates.map((template) => {
    const start = new Date();
    start.setHours(0, 0, 0, 0);
    start.setDate(start.getDate() + template.dayOffset);
    start.setHours(template.hour, template.minute, 0, 0);
    const end = new Date(start.getTime() + template.durationMinutes * 60 * 1000);

    return {
      id: template.id,
      roomId: template.roomId,
      roomName: template.roomName,
      category: template.category,
      startDateTime: start.toISOString(),
      endDateTime: end.toISOString(),
      status: template.status,
      notes: template.notes,
    };
  });
};

export function AppDataProvider({ children }) {
  // Demo-Daten werden zentral verwaltet, sodass jede Seite über den Context darauf zugreifen kann.
  const [favorites, setFavorites] = useState(INITIAL_FAVORITES);
  const [bookings, setBookings] = useState(createInitialBookings);

  const toggleFavorite = useCallback((roomId) => {
    setFavorites((prev) =>
      prev.includes(roomId)
        ? prev.filter((id) => id !== roomId)
        : [...prev, roomId],
    );
  }, []);

  const cancelBooking = useCallback((bookingId) => {
    setBookings((prev) => prev.filter((booking) => booking.id !== bookingId));
  }, []);

  const value = useMemo(
    () => ({
      favorites,
      toggleFavorite,
      bookings,
      setBookings,
      cancelBooking,
    }),
    [favorites, toggleFavorite, bookings, setBookings, cancelBooking],
  );

  return <AppDataContext.Provider value={value}>{children}</AppDataContext.Provider>;
}

export function useAppData() {
  // Hilfs-Hook kapselt den Zugriff und stellt sicher, dass der Provider verwendet wird.
  const context = useContext(AppDataContext);
  if (!context) {
    throw new Error("useAppData must be used within an AppDataContext provider");
  }
  return context;
}
