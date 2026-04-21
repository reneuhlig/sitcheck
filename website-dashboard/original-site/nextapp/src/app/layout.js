"use client";

import { useState } from "react";
import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { AppDataProvider, useAppData } from "@/context/AppDataContext";
import logo from "./logo.png";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

const APP_NAV_ITEMS = [
  { href: "/", label: "Dashboard" },
  { href: "/bookings", label: "Buchungen" },
];

const PORTAL_NAV_ITEMS = [
  { href: "/realtime", label: "Videomonitor" },
  { href: "/admin-bookings", label: "Buchungsmonitor", adminOnly: true },
];

function Header() {
  const pathname = usePathname();
  const { authenticated, user, logout, sessionLoading } = useAppData();
  const [logoutPending, setLogoutPending] = useState(false);
  const [adminNoticeOpen, setAdminNoticeOpen] = useState(false);

  async function handleLogout() {
    setLogoutPending(true);
    try {
      await logout();
    } finally {
      setLogoutPending(false);
    }
  }

  function openAdminNotice() {
    setAdminNoticeOpen(true);
  }

  return (
    <header className="sticky top-0 z-30 border-b border-white/60 bg-[color:var(--surface-frost)] backdrop-blur-xl">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-5 px-4 py-4 sm:px-6 lg:px-8">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-4">
            <div className="overflow-hidden rounded-3xl border border-[color:var(--accent)] bg-[color:var(--accent)] p-1 shadow-[0_12px_28px_rgba(139,198,236,0.38)]">
              <Image
                src={logo}
                alt="Sitcheck Logo"
                width={80}
                height={80}
                priority
                className="h-16 w-16 scale-[1.55] -translate-y-[5px] rounded-[1.35rem] object-cover sm:h-20 sm:w-20"
              />
            </div>
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.24em] text-[color:var(--accent-strong)]">
                Unibibliothek Live
              </p>
              <h1 className="text-xl font-semibold text-[color:var(--ink-strong)] sm:text-2xl">
                Sitcheck
              </h1>
            </div>
          </div>

          <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
            <div className="rounded-3xl border border-white/70 bg-white/90 px-4 py-3 text-sm shadow-[0_20px_45px_rgba(15,23,42,0.08)]">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[color:var(--ink-muted)]">
                Zugang
              </p>
              {sessionLoading ? (
                <p className="mt-1 text-[color:var(--ink-soft)]">Sitzung wird geladen …</p>
              ) : authenticated ? (
                <div className="mt-1 flex items-center gap-3">
                  <div>
                    <p className="font-semibold text-[color:var(--ink-strong)]">{user?.username}</p>
                    <p className="text-xs text-[color:var(--ink-soft)]">Eingeloggt für Buchungen</p>
                  </div>
                  <button
                    type="button"
                    onClick={handleLogout}
                    disabled={logoutPending}
                    className="rounded-full border border-[color:var(--stroke-strong)] px-3 py-1 text-xs font-semibold text-[color:var(--ink-strong)] transition hover:border-[color:var(--accent)] hover:text-[color:var(--accent-strong)] disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {logoutPending ? "Abmelden …" : "Abmelden"}
                  </button>
                </div>
              ) : (
                <Link
                  href="/login"
                  className="mt-1 inline-flex items-center gap-2 text-sm font-semibold text-[color:var(--accent-strong)] transition hover:text-[color:var(--accent)]"
                >
                  Anmelden oder registrieren
                  <span aria-hidden="true">→</span>
                </Link>
              )}
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <nav className="flex flex-wrap gap-2">
            {APP_NAV_ITEMS.map((item) => {
              const active = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
                    active
                      ? "bg-[color:var(--accent)] text-white shadow-[0_12px_28px_rgba(139,198,236,0.38)]"
                      : "border border-white/70 bg-white/85 text-[color:var(--ink-strong)] hover:border-[color:var(--accent-soft)] hover:text-[color:var(--accent-strong)]"
                  }`}
                >
                  {item.label}
                </Link>
              );
            })}
            {!authenticated && !sessionLoading && (
              <Link
                href="/login"
                className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
                  pathname === "/login"
                    ? "bg-[color:var(--accent)] text-white shadow-[0_12px_28px_rgba(139,198,236,0.38)]"
                    : "border border-white/70 bg-white/85 text-[color:var(--ink-strong)] hover:border-[color:var(--accent-soft)] hover:text-[color:var(--accent-strong)]"
                }`}
              >
                Login
              </Link>
            )}
          </nav>

          <div className="flex flex-wrap gap-2">
            {PORTAL_NAV_ITEMS.map((item) => {
              const active = pathname === item.href;
              const adminBlocked = item.adminOnly && user?.role !== "admin";
              const classes = `rounded-full px-4 py-2 text-sm font-semibold transition ${
                active && !adminBlocked
                  ? "bg-[color:var(--accent)] text-white shadow-[0_12px_28px_rgba(139,198,236,0.38)]"
                  : "border border-[color:var(--stroke-strong)] bg-[color:var(--surface-raised)] text-[color:var(--ink-strong)] hover:border-[color:var(--accent-soft)] hover:text-[color:var(--accent-strong)]"
              }`;

              if (adminBlocked) {
                return (
                  <button
                    key={item.href}
                    type="button"
                    onClick={openAdminNotice}
                    className={classes}
                  >
                    {item.label}
                  </button>
                );
              }

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={classes}
                >
                  {item.label}
                </Link>
              );
            })}
          </div>
        </div>
      </div>
      {adminNoticeOpen && (
        <div
          className="fixed inset-0 z-[60] flex min-h-dvh items-center justify-center overflow-y-auto bg-slate-950/35 px-4 py-6 sm:px-6"
          role="dialog"
          aria-modal="true"
        >
          <div className="my-auto w-full max-w-md rounded-[2rem] border border-white/70 bg-white p-6 shadow-[0_30px_80px_rgba(15,23,42,0.22)]">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[color:var(--ink-muted)]">
              Zugriff eingeschränkt
            </p>
            <h2 className="mt-2 text-2xl font-semibold text-[color:var(--ink-strong)]">
              Buchungsmonitor ist nur für Admins verfügbar
            </h2>
            <p className="mt-4 text-sm leading-7 text-[color:var(--ink-soft)]">
              {authenticated
                ? "Dein aktueller Account hat keine Admin-Rechte. Melde dich mit einem Admin-Konto an, um die Buchungen aller Nutzer zu sehen."
                : "Melde dich mit einem Admin-Konto an, um die Buchungen aller Nutzer zu sehen."}
            </p>
            <div className="mt-6 flex flex-wrap gap-3">
              {!authenticated && (
                <Link
                  href="/login"
                  onClick={() => setAdminNoticeOpen(false)}
                  className="inline-flex items-center gap-2 rounded-full bg-[color:var(--accent)] px-4 py-2 text-sm font-semibold text-white transition hover:bg-[color:var(--accent-strong)]"
                >
                  Zum Login
                  <span aria-hidden="true">→</span>
                </Link>
              )}
              <button
                type="button"
                onClick={() => setAdminNoticeOpen(false)}
                className="rounded-full border border-[color:var(--stroke-strong)] px-4 py-2 text-sm font-semibold text-[color:var(--ink-strong)] transition hover:border-[color:var(--accent-soft)] hover:text-[color:var(--accent-strong)]"
              >
                Schließen
              </button>
            </div>
          </div>
        </div>
      )}
    </header>
  );
}

function AppChrome({ children }) {
  return (
    <div className="relative min-h-screen overflow-hidden">
      <div className="pointer-events-none absolute inset-x-0 top-0 h-80 bg-[radial-gradient(circle_at_top_left,rgba(188,228,254,0.55),transparent_42%),radial-gradient(circle_at_top_right,rgba(139,198,236,0.28),transparent_35%)]" />
      <Header />
      <main className="relative z-10 mx-auto flex w-full max-w-7xl flex-1 flex-col px-4 py-8 sm:px-6 lg:px-8">
        {children}
      </main>
      <footer className="relative z-10 border-t border-white/60 bg-white/55 px-4 py-6 backdrop-blur sm:px-6 lg:px-8">
        <div className="mx-auto flex w-full max-w-7xl flex-col gap-2 text-sm text-[color:var(--ink-soft)] sm:flex-row sm:items-center sm:justify-between">
          <p className="font-medium text-[color:var(--ink-muted)]">DHBW Mannheim · Sitcheck Portal</p>
        </div>
      </footer>
    </div>
  );
}

export default function RootLayout({ children }) {
  return (
    <html lang="de" className="scroll-smooth">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <AppDataProvider>
          <AppChrome>{children}</AppChrome>
        </AppDataProvider>
      </body>
    </html>
  );
}
