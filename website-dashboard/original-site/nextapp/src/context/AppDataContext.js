"use client";

import { createContext, startTransition, useContext, useEffect, useRef, useState } from "react";
import { sitcheckApi } from "@/lib/api";

export const AppDataContext = createContext(null);

export function AppDataProvider({ children }) {
  const [sessionLoading, setSessionLoading] = useState(true);
  const [sessionError, setSessionError] = useState("");
  const [authenticated, setAuthenticated] = useState(false);
  const [user, setUser] = useState(null);

  const [bookingsLoading, setBookingsLoading] = useState(false);
  const [bookingsError, setBookingsError] = useState("");
  const [bookings, setBookings] = useState([]);
  const [bookingRevision, setBookingRevision] = useState(0);

  async function refreshBookings(signal, options = {}) {
    const { force = false } = options;
    if (!force && !authenticated) {
      startTransition(() => {
        setBookings([]);
        setBookingsError("");
      });
      return [];
    }

    setBookingsLoading(true);
    try {
      const bookingItems = await sitcheckApi.getBookings(signal);
      startTransition(() => {
        setBookings(Array.isArray(bookingItems) ? bookingItems : []);
        setBookingsError("");
      });
      return bookingItems;
    } catch (error) {
      const message = error instanceof Error ? error.message : "Buchungen konnten nicht geladen werden.";
      startTransition(() => setBookingsError(message));
      throw error;
    } finally {
      setBookingsLoading(false);
    }
  }

  async function refreshSession(signal) {
    setSessionLoading(true);
    try {
      const payload = await sitcheckApi.authMe(signal);
      const isAuthenticated = Boolean(payload?.authenticated);
      startTransition(() => {
        setAuthenticated(isAuthenticated);
        setUser(isAuthenticated ? payload.user ?? null : null);
        setSessionError("");
      });

      if (isAuthenticated) {
        await refreshBookings(signal, { force: true });
      } else {
        startTransition(() => {
          setBookings([]);
          setBookingsError("");
        });
      }
      return payload;
    } catch (error) {
      const message = error instanceof Error ? error.message : "Sitzung konnte nicht geprüft werden.";
      startTransition(() => {
        setAuthenticated(false);
        setUser(null);
        setBookings([]);
        setSessionError(message);
      });
      throw error;
    } finally {
      setSessionLoading(false);
    }
  }

  const refreshSessionRef = useRef(refreshSession);
  refreshSessionRef.current = refreshSession;

  useEffect(() => {
    const abortController = new AbortController();
    refreshSessionRef.current(abortController.signal).catch(() => {});
    return () => abortController.abort();
  }, []);

  async function register(credentials) {
    const payload = await sitcheckApi.register(credentials);
    startTransition(() => {
      setAuthenticated(Boolean(payload?.authenticated));
      setUser(payload?.user ?? null);
      setSessionError("");
    });
    await refreshBookings(undefined, { force: true });
    return payload;
  }

  async function login(credentials) {
    const payload = await sitcheckApi.login(credentials);
    startTransition(() => {
      setAuthenticated(Boolean(payload?.authenticated));
      setUser(payload?.user ?? null);
      setSessionError("");
    });
    await refreshBookings(undefined, { force: true });
    return payload;
  }

  async function logout() {
    await sitcheckApi.logout();
    startTransition(() => {
      setAuthenticated(false);
      setUser(null);
      setBookings([]);
      setBookingsError("");
    });
  }

  async function createBooking(payload) {
    const created = await sitcheckApi.createBooking(payload);
    await refreshBookings();
    startTransition(() => setBookingRevision((current) => current + 1));
    return created;
  }

  async function cancelBooking(bookingId) {
    const cancelled = await sitcheckApi.cancelBooking(bookingId);
    await refreshBookings();
    startTransition(() => setBookingRevision((current) => current + 1));
    return cancelled;
  }

  return (
    <AppDataContext.Provider
      value={{
        sessionLoading,
        sessionError,
        authenticated,
        user,
        refreshSession,
        register,
        login,
        logout,
        bookingsLoading,
        bookingsError,
        bookings,
        refreshBookings,
        createBooking,
        cancelBooking,
        bookingRevision,
      }}
    >
      {children}
    </AppDataContext.Provider>
  );
}

export function useAppData() {
  const context = useContext(AppDataContext);
  if (!context) {
    throw new Error("useAppData must be used within an AppDataContext provider");
  }
  return context;
}
