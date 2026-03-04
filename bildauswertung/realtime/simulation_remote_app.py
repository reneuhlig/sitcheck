#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, Optional, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request

from flask import Flask, jsonify, render_template_string, request


HTML_PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Simulation Remote Control</title>
  <style>
    body { font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; }
    .wrap { max-width: 900px; margin: 24px auto; padding: 0 16px; }
    .panel { background: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 14px; }
    .row { display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0; align-items: center; }
    button, select { background: #1f2937; color: #e5e7eb; border: 1px solid #374151; border-radius: 6px; padding: 8px 10px; }
    button:hover, select:hover { background: #374151; }
    .muted { color: #9ca3af; font-size: 13px; }
    #clip_select { min-width: 520px; max-width: 100%; }
    #status { margin-top: 12px; }
  </style>
</head>
<body>
  <div class="wrap">
    <h2>Simulation Remote Control</h2>
    <div class="panel">
      <div class="row">
        <label>Control Mode</label>
        <select id="control_mode">
          <option value="remote_control">remote_control</option>
          <option value="auto_rules">auto_rules (später)</option>
        </select>
      </div>
      <div class="row">
        <label>Clip</label>
        <select id="clip_select"></select>
      </div>
      <div class="row">
        <button id="reload_btn">Catalog Reload</button>
        <button id="apply_btn">Play Once</button>
      </div>
      <div id="catalog_info" class="muted">Catalog lädt...</div>
      <div id="status" class="muted">Bereit.</div>
    </div>
  </div>

  <script>
    const clipSelect = document.getElementById('clip_select');
    const controlModeEl = document.getElementById('control_mode');
    const catalogInfoEl = document.getElementById('catalog_info');
    const statusEl = document.getElementById('status');

    async function loadCatalog() {
      const res = await fetch('/api/catalog');
      const payload = await res.json();
      if (!payload.ok) {
        catalogInfoEl.textContent = 'Catalog konnte nicht geladen werden: ' + (payload.error || 'unknown');
        return;
      }

      const clips = Array.isArray(payload.catalog?.clips) ? payload.catalog.clips : [];
      clipSelect.innerHTML = '';
      for (const clip of clips) {
        const option = document.createElement('option');
        option.value = String(clip.clip_id || '');
        const aliases = Number(clip.aliases_count || 0);
        option.textContent = aliases > 0
          ? `${clip.display_name || clip.clip_id} (${clip.relative_path}) [dup:${aliases}]`
          : `${clip.display_name || clip.clip_id} (${clip.relative_path})`;
        clipSelect.appendChild(option);
      }

      const current = payload.video_source || {};
      const selectedId = String(current.simulation_clip_id || '');
      if (selectedId && Array.from(clipSelect.options).some((opt) => opt.value === selectedId)) {
        clipSelect.value = selectedId;
      }
      const mode = String(current.simulation_control_mode || 'remote_control');
      controlModeEl.value = mode === 'auto_rules' ? 'auto_rules' : 'remote_control';
      const basicClip = String(current.simulation_basic_clip_id || '');
      const pendingCount = Number(current.simulation_pending_count || 0);

      const summary = payload.catalog?.summary || {};
      catalogInfoEl.textContent = `Clips: ${summary.clips || clips.length || 0} | Duplikate: ${summary.duplicates || 0}`;
      statusEl.textContent = current.active_source
        ? `Aktiv: ${current.active_source} | Basic: ${basicClip || '-'} | Queue: ${pendingCount}`
        : 'Keine aktive Quelle';
    }

    async function applyClip() {
      const clipId = String(clipSelect.value || '').trim();
      if (!clipId) {
        statusEl.textContent = 'Kein Clip ausgewählt';
        return;
      }

      const controlMode = controlModeEl.value === 'auto_rules' ? 'auto_rules' : 'remote_control';
      const res = await fetch('/api/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ clip_id: clipId, control_mode: controlMode }),
      });
      const payload = await res.json();
      if (!payload.ok) {
        statusEl.textContent = 'Umschalten fehlgeschlagen: ' + (payload.error || 'unknown');
        return;
      }

      const selected = payload.selected || {};
      const next = payload.next || {};
      statusEl.textContent = `Play-once gestartet: ${selected.display_name || clipId} -> danach ${next.display_name || 'basic loop'}`;
      await loadCatalog();
    }

    document.getElementById('reload_btn').addEventListener('click', loadCatalog);
    document.getElementById('apply_btn').addEventListener('click', applyClip);

    loadCatalog();
  </script>
</body>
</html>
"""


def _http_json_request(
    base_url: str,
    path: str,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    timeout_sec: float = 3.0,
) -> Tuple[int, Dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib_request.Request(url=url, data=data, headers=headers, method=method)
    try:
        with urllib_request.urlopen(req, timeout=timeout_sec) as response:
            raw = response.read().decode("utf-8")
            parsed = json.loads(raw) if raw else {}
            return int(getattr(response, "status", 200) or 200), parsed
    except urllib_error.HTTPError as exc:
        try:
            raw = exc.read().decode("utf-8")
            parsed = json.loads(raw) if raw else {}
        except Exception:
            parsed = {"ok": False, "error": str(exc)}
        return int(exc.code or 500), parsed
    except Exception as exc:
        return 502, {"ok": False, "error": f"Dashboard API nicht erreichbar: {exc}"}


def create_app(dashboard_base_url: str) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(HTML_PAGE)

    @app.route("/api/catalog", methods=["GET"])
    def api_catalog():
        status_catalog, catalog = _http_json_request(dashboard_base_url, "/api/simulation/catalog", method="GET")
        status_source, source = _http_json_request(dashboard_base_url, "/api/video-source", method="GET")

        if status_catalog >= 400:
            return jsonify({"ok": False, "error": catalog.get("error", "catalog error")}), status_catalog
        if status_source >= 400:
            return jsonify({"ok": False, "error": source.get("error", "video source error")}), status_source

        return jsonify({"ok": True, "catalog": catalog, "video_source": source})

    @app.route("/api/select", methods=["POST"])
    def api_select():
        payload = request.get_json(silent=True) or {}
        status_code, data = _http_json_request(
            dashboard_base_url,
            "/api/simulation/select",
            method="POST",
            payload=payload,
        )
        return jsonify(data), status_code

    @app.route("/health")
    def health():
        return jsonify({"ok": True, "dashboard_base_url": dashboard_base_url})

    return app


def main():
    parser = argparse.ArgumentParser(description="Simulation remote control app")
    parser.add_argument("--host", default=os.getenv("SITCHECK_SIM_REMOTE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("SITCHECK_SIM_REMOTE_PORT", "8091")))
    parser.add_argument(
        "--dashboard-base-url",
        default=os.getenv("SITCHECK_DASHBOARD_API_BASE", "http://127.0.0.1:8080"),
    )
    args = parser.parse_args()

    app = create_app(dashboard_base_url=args.dashboard_base_url)
    app.run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
