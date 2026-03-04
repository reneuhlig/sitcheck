# Local Overrides

Dieses Verzeichnis enthält lokale Anpassungen, die nach einem Upstream-Sync
gezielt wieder auf `nextapp/` angewendet werden.

Ablauf:
1. Upstream nach `nextapp/` synchronisieren
2. `scripts/apply_local_overrides.sh` ausführen
3. Statische Seite neu bauen

Aktive Overrides:
- `src/app/layout.js`
- `src/app/page.js`
