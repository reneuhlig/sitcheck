"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useAppData } from "@/context/AppDataContext";

export default function LoginPage() {
  const router = useRouter();
  const { authenticated, user, register, login, logout, sessionLoading } = useAppData();
  const [mode, setMode] = useState("login");
  const [formState, setFormState] = useState({ username: "", password: "", role: "user" });
  const [pending, setPending] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  async function handleSubmit(event) {
    event.preventDefault();
    setPending(true);
    setErrorMessage("");
    try {
      const credentials = {
        username: formState.username,
        password: formState.password,
      };
      if (mode === "register") {
        await register({ ...credentials, role: formState.role });
      } else {
        await login(credentials);
      }
      router.push("/bookings");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Anmeldung fehlgeschlagen.";
      setErrorMessage(message);
    } finally {
      setPending(false);
    }
  }

  async function handleLogout() {
    setPending(true);
    setErrorMessage("");
    try {
      await logout();
    } finally {
      setPending(false);
    }
  }

  if (sessionLoading) {
    return (
      <div className="rounded-[2rem] border border-[color:var(--stroke-soft)] bg-[color:var(--surface-raised)] p-8 text-sm text-[color:var(--ink-soft)] shadow-[0_24px_54px_rgba(15,23,42,0.08)]">
        Sitzung wird geladen …
      </div>
    );
  }

  if (authenticated) {
    return (
      <section className="grid gap-6 xl:grid-cols-[1fr_0.9fr]">
        <article className="rounded-[2rem] border border-[color:var(--stroke-soft)] bg-[color:var(--surface-raised)] p-8 shadow-[0_24px_54px_rgba(15,23,42,0.08)]">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[color:var(--ink-muted)]">
            Bereits angemeldet
          </p>
          <h2 className="mt-2 text-3xl font-semibold text-[color:var(--ink-strong)]">
            Willkommen zurück, {user?.username}
          </h2>
          <p className="mt-4 max-w-2xl text-sm leading-7 text-[color:var(--ink-soft)]">
            Dein Account ist aktiv. Du kannst jetzt Buchungen anlegen, bestehende Reservierungen
            verwalten und direkt sehen, wie sich deine Slots auf die Kurzfristprognose auswirken.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Link
              href="/bookings"
              className="inline-flex items-center gap-2 rounded-full bg-[color:var(--accent)] px-4 py-2 text-sm font-semibold text-white transition hover:bg-[color:var(--accent-strong)]"
            >
              Zu meinen Buchungen
              <span aria-hidden="true">→</span>
            </Link>
            <button
              type="button"
              onClick={handleLogout}
              disabled={pending}
              className="rounded-full border border-[color:var(--stroke-strong)] px-4 py-2 text-sm font-semibold text-[color:var(--ink-strong)] transition hover:border-[color:var(--accent-soft)] hover:text-[color:var(--accent-strong)] disabled:cursor-not-allowed disabled:opacity-60"
            >
              {pending ? "Abmelden …" : "Abmelden"}
            </button>
          </div>
        </article>

        <article className="rounded-[2rem] border border-[color:var(--stroke-soft)] bg-[color:var(--surface-raised)] p-8 shadow-[0_24px_54px_rgba(15,23,42,0.08)]">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[color:var(--ink-muted)]">
            Was dein Konto kann
          </p>
          <ul className="mt-4 space-y-3 text-sm leading-7 text-[color:var(--ink-soft)]">
            <li>Bibliotheks-Slots direkt im Backend anlegen und stornieren.</li>
            <li>Forecast-Overlay automatisch für bestätigte Buchungen erhalten.</li>
            <li>Ohne zusätzliche Verifizierung arbeiten, aber mit sicher gehashten Passwörtern.</li>
          </ul>
        </article>
      </section>
    );
  }

  return (
    <section className="grid gap-6 xl:grid-cols-[0.92fr_1.08fr]">
      <article className="rounded-[2rem] border border-[color:var(--stroke-soft)] bg-[color:var(--surface-raised)] p-8 shadow-[0_24px_54px_rgba(15,23,42,0.08)]">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[color:var(--ink-muted)]">
          Konto für Buchungen
        </p>
        <h2 className="mt-2 text-3xl font-semibold text-[color:var(--ink-strong)]">
          Login oder Registrierung
        </h2>
        <p className="mt-4 text-sm leading-7 text-[color:var(--ink-soft)]">
          Sitcheck nutzt für das MVP einen einfachen Username-/Passwort-Login ohne Verifizierung.
          Passwörter werden im Backend nicht im Klartext gespeichert.
        </p>
        <div className="mt-6 flex gap-3">
          {[
            { id: "login", label: "Anmelden" },
            { id: "register", label: "Registrieren" },
          ].map((item) => {
            const active = mode === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => setMode(item.id)}
                className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
                  active
                    ? "bg-[color:var(--accent)] text-white"
                    : "border border-[color:var(--stroke-strong)] text-[color:var(--ink-strong)] hover:border-[color:var(--accent-soft)] hover:text-[color:var(--accent-strong)]"
                }`}
              >
                {item.label}
              </button>
            );
          })}
        </div>

        <form onSubmit={handleSubmit} className="mt-8 space-y-5">
          <label className="block">
            <span className="text-xs font-semibold uppercase tracking-[0.16em] text-[color:var(--ink-muted)]">
              Username
            </span>
            <input
              type="text"
              value={formState.username}
              onChange={(event) => setFormState((current) => ({ ...current, username: event.target.value }))}
              className="mt-2 w-full rounded-[1.25rem] border border-[color:var(--stroke-strong)] bg-white px-4 py-3 text-[color:var(--ink-strong)] outline-none focus:border-[color:var(--accent)]"
              placeholder="z. B. maxmustermann"
              required
              minLength={3}
            />
          </label>

          <label className="block">
            <span className="text-xs font-semibold uppercase tracking-[0.16em] text-[color:var(--ink-muted)]">
              Passwort
            </span>
            <input
              type="password"
              value={formState.password}
              onChange={(event) => setFormState((current) => ({ ...current, password: event.target.value }))}
              className="mt-2 w-full rounded-[1.25rem] border border-[color:var(--stroke-strong)] bg-white px-4 py-3 text-[color:var(--ink-strong)] outline-none focus:border-[color:var(--accent)]"
              placeholder="Mindestens 4 Zeichen"
              required
              minLength={4}
            />
          </label>

          {mode === "register" && (
            <fieldset>
              <legend className="text-xs font-semibold uppercase tracking-[0.16em] text-[color:var(--ink-muted)]">
                Rolle
              </legend>
              <div className="mt-2 grid gap-3 sm:grid-cols-2">
                {[
                  {
                    id: "user",
                    label: "Normaler User",
                    description: "Standardzugang für Login und Buchungen.",
                  },
                  {
                    id: "admin",
                    label: "Admin",
                    description: "Account wird direkt mit Admin-Rechten gespeichert.",
                  },
                ].map((item) => {
                  const active = formState.role === item.id;
                  return (
                    <label
                      key={item.id}
                      className={`block cursor-pointer rounded-[1.25rem] border px-4 py-3 transition ${
                        active
                          ? "border-[color:var(--accent)] bg-[color:var(--surface-muted)]"
                          : "border-[color:var(--stroke-strong)] bg-white hover:border-[color:var(--accent-soft)]"
                      }`}
                    >
                      <input
                        type="radio"
                        name="role"
                        value={item.id}
                        checked={active}
                        onChange={(event) => setFormState((current) => ({ ...current, role: event.target.value }))}
                        className="sr-only"
                      />
                      <span className="block text-sm font-semibold text-[color:var(--ink-strong)]">{item.label}</span>
                      <span className="mt-1 block text-xs leading-6 text-[color:var(--ink-soft)]">
                        {item.description}
                      </span>
                    </label>
                  );
                })}
              </div>
            </fieldset>
          )}

          {errorMessage && (
            <div className="rounded-[1.5rem] border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
              {errorMessage}
            </div>
          )}

          <button
            type="submit"
            disabled={pending}
            className="inline-flex items-center gap-2 rounded-full bg-[color:var(--accent)] px-5 py-3 text-sm font-semibold text-white transition hover:bg-[color:var(--accent-strong)] disabled:cursor-not-allowed disabled:opacity-60"
          >
            {pending ? "Bitte warten …" : mode === "register" ? "Konto erstellen" : "Anmelden"}
          </button>
        </form>
      </article>

      <article className="rounded-[2rem] border border-[color:var(--stroke-soft)] bg-[color:var(--surface-raised)] p-8 shadow-[0_24px_54px_rgba(15,23,42,0.08)]">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[color:var(--ink-muted)]">
          Nach dem Login
        </p>
        <h3 className="mt-2 text-3xl font-semibold text-[color:var(--ink-strong)]">
          Vom Account direkt in den Forecast
        </h3>
        <div className="mt-6 grid gap-4">
          <div className="rounded-[1.5rem] bg-[color:var(--surface-muted)] px-4 py-4">
            <p className="text-sm font-semibold text-[color:var(--ink-strong)]">1. Bibliotheks-Slot anlegen</p>
            <p className="mt-2 text-sm leading-7 text-[color:var(--ink-soft)]">
              Du wählst Start und Ende deines Aufenthalts direkt in der Buchungsansicht.
            </p>
          </div>
          <div className="rounded-[1.5rem] bg-[color:var(--surface-muted)] px-4 py-4">
            <p className="text-sm font-semibold text-[color:var(--ink-strong)]">2. Backend speichert sofort</p>
            <p className="mt-2 text-sm leading-7 text-[color:var(--ink-soft)]">
              Die Reservierung wird ohne weiteren Verifizierungs- oder Approval-Flow bestätigt.
            </p>
          </div>
          <div className="rounded-[1.5rem] bg-[color:var(--surface-muted)] px-4 py-4">
            <p className="text-sm font-semibold text-[color:var(--ink-strong)]">3. Prognose passt sich an</p>
            <p className="mt-2 text-sm leading-7 text-[color:var(--ink-soft)]">
              Jede bestätigte Buchung erhöht die 60-Minuten-Prognose im gebuchten Zeitraum um genau einen Platz.
            </p>
          </div>
        </div>
        <Link
          href="/"
          className="mt-6 inline-flex items-center gap-2 text-sm font-semibold text-[color:var(--accent-strong)] transition hover:text-[color:var(--accent)]"
        >
          Zurück zum Dashboard
          <span aria-hidden="true">→</span>
        </Link>
      </article>
    </section>
  );
}
