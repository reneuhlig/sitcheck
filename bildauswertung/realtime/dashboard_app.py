#!/usr/bin/env python3
import argparse
from collections import deque
from datetime import datetime
import os
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request, send_from_directory

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
BILDAUSWERTUNG_DIR = os.path.join(WORKSPACE_ROOT, "bildauswertung")

if BILDAUSWERTUNG_DIR not in sys.path:
    sys.path.insert(0, BILDAUSWERTUNG_DIR)

from ConfigManager import ConfigManager
from DetectorFactory import build_person_detector
from integration.prognose_db_writer import PrognoseDbWriter
from integration.profile_occupancy_simulation import ProfileOccupancySimulation
from OccupancyStateModule import OccupancyStateModule
from TrajectoryEntryAnalysisModule import EntranceZoneConfig, TrajectoryEntryAnalysisModule
from VideoInputModule import VideoInputModule
from VisualizationOutputModule import VisualizationOutputModule
from YOLOTrackingModule import YOLOTrackingModule
from simulation_video_registry import DEFAULT_SIMULATION_DIR, SimulationVideoRegistry


HTML_PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Sitcheck Dashboard</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

    :root {
      --bg-0: #06120f;
      --bg-1: #0b1f1a;
      --bg-2: #102720;
      --panel: rgba(12, 28, 24, 0.76);
      --panel-border: rgba(119, 193, 169, 0.24);
      --text-0: #e7fff6;
      --text-1: #bbddd1;
      --accent: #5ae4b8;
      --accent-2: #e3ff6a;
      --warn: #ffd166;
      --danger: #ff6b6b;
    }

    * { box-sizing: border-box; }
    html, body { min-height: 100%; }
    body {
      font-family: 'Sora', 'Segoe UI', sans-serif;
      color: var(--text-0);
      margin: 0;
      background:
        radial-gradient(1200px 580px at 10% -10%, #1f5948 0%, rgba(31, 89, 72, 0.1) 55%, transparent 72%),
        radial-gradient(900px 520px at 100% 0%, #5b7b1e 0%, rgba(91, 123, 30, 0.08) 45%, transparent 70%),
        linear-gradient(135deg, var(--bg-0), var(--bg-1) 48%, var(--bg-2));
      padding: 18px;
    }

    .wrap {
      width: min(1820px, 100%);
      margin: 0 auto;
      padding: 8px;
    }

    .header {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: flex-end;
      margin-bottom: 14px;
    }
    .header h2 {
      margin: 0;
      font-size: clamp(22px, 2.4vw, 34px);
      letter-spacing: 0.02em;
      font-weight: 800;
    }
    .subtitle {
      margin-top: 5px;
      color: var(--text-1);
      font-size: 13px;
    }

    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1.9fr) minmax(330px, 1fr);
      gap: 16px;
      align-items: start;
    }

    .panel {
      background: var(--panel);
      backdrop-filter: blur(10px);
      border: 1px solid var(--panel-border);
      border-radius: 16px;
      padding: 14px;
      box-shadow: 0 18px 48px rgba(0, 0, 0, 0.35);
    }

    .controls-panel { max-height: calc(100vh - 70px); overflow: auto; }
    .controls-panel::-webkit-scrollbar { width: 10px; }
    .controls-panel::-webkit-scrollbar-thumb { background: #2a5f50; border-radius: 10px; }

    .title {
      margin: 0 0 10px;
      font-size: 13px;
      letter-spacing: 0.09em;
      text-transform: uppercase;
      color: var(--text-1);
      font-weight: 700;
    }

    .feed-wrap {
      position: relative;
      width: 100%;
      aspect-ratio: 16 / 9;
      min-height: 420px;
      background: #040906;
      overflow: hidden;
      border-radius: 14px;
      border: 1px solid rgba(114, 200, 173, 0.4);
      box-shadow: inset 0 0 0 1px rgba(90, 228, 184, 0.2), 0 16px 36px rgba(0, 0, 0, 0.5);
    }

    #dashFeed { width: 100%; height: 100%; object-fit: contain; display: block; }
    #dashFeed { background: #000; }
    #overlay { position: absolute; inset: 0; pointer-events: auto; }

    .row { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0; }

    .stat {
      background: rgba(8, 23, 18, 0.88);
      border: 1px solid rgba(123, 200, 174, 0.24);
      border-radius: 12px;
      padding: 9px;
      flex: 1 1 96px;
      min-width: 94px;
    }

    .stat b {
      display: block;
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #9ec9bc;
      margin-bottom: 6px;
    }

    .stat > div {
      font-family: 'JetBrains Mono', monospace;
      font-size: 18px;
      font-weight: 600;
      color: #f4fff9;
    }

    button, select, input {
      background: #17362d;
      color: #ecfff9;
      border: 1px solid #3f7a68;
      border-radius: 10px;
      padding: 9px 11px;
      font-family: inherit;
      transition: 160ms ease;
    }
    button, select { cursor: pointer; }
    button:hover, select:hover { background: #21473c; transform: translateY(-1px); }
    input:focus, select:focus, button:focus { outline: 2px solid rgba(90, 228, 184, 0.46); outline-offset: 1px; }

    #toggle_analysis {
      background: linear-gradient(110deg, #4ec89f, #3dad86);
      border-color: rgba(192, 255, 231, 0.4);
      color: #062117;
      font-weight: 700;
      box-shadow: 0 8px 16px rgba(32, 86, 69, 0.5);
    }
    .det-btn.active {
      background: linear-gradient(110deg, #4ec89f, #3dad86);
      border-color: rgba(192, 255, 231, 0.4);
      color: #062117;
      font-weight: 700;
    }

    .navlink {
      display: inline-block;
      background: #153328;
      color: #dcfff3;
      border: 1px solid #3e7f69;
      border-radius: 10px;
      padding: 8px 11px;
      text-decoration: none;
      transition: 160ms ease;
      font-size: 13px;
    }
    .navlink:hover { background: #1e4638; transform: translateY(-1px); }

    .muted { color: #9ec6ba; font-size: 13px; }
    .ok { color: var(--accent); }
    .warn { color: var(--warn); }

    #status {
      margin-top: 8px;
      padding: 8px 10px;
      border-radius: 10px;
      background: rgba(8, 25, 20, 0.7);
      border: 1px solid rgba(98, 177, 151, 0.2);
    }

    #stream_status {
      margin-top: 8px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
    }

    .sim-queue {
      margin-top: 12px;
      border: 1px solid rgba(123, 200, 174, 0.24);
      border-radius: 12px;
      padding: 10px;
      background: rgba(7, 19, 16, 0.72);
    }

    .queue-board {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
      min-height: 34px;
      align-items: center;
    }

    .queue-pill {
      padding: 5px 8px;
      border-radius: 999px;
      border: 1px solid #3f7a68;
      background: #17362d;
      color: #ddfff5;
      font-size: 12px;
      line-height: 1.2;
      white-space: nowrap;
      max-width: 280px;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .queue-pill.active {
      background: linear-gradient(110deg, #4ec89f, #2f8f6f);
      color: #062117;
      border-color: rgba(199, 255, 235, 0.4);
      font-weight: 700;
    }

    .queue-pill.basic {
      border-style: dashed;
      opacity: 0.88;
    }

    .sim-select {
      flex: 1 1 100%;
      min-width: 240px;
    }

    @media (max-width: 1280px) {
      .grid { grid-template-columns: minmax(0, 1fr); }
      .controls-panel { max-height: none; overflow: visible; }
      .feed-wrap { min-height: 320px; }
    }

    @media (max-width: 720px) {
      body { padding: 10px; }
      .panel { padding: 11px; border-radius: 13px; }
      .header { flex-direction: column; align-items: flex-start; }
      .stat > div { font-size: 16px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <div>
        <h2>Sitcheck Control Deck</h2>
        <div class="subtitle">Livefeed, Dual-Polygon-Auswertung und Simulations-Fernsteuerung in einer Oberfläche</div>
      </div>
    </div>
    <div class="grid">
      <div class="panel controls-panel">
        <h3 class="title">Live Feed + Tracking Overlay</h3>
        <div class="feed-wrap">
          <video id="dashFeed" muted autoplay playsinline></video>
          <canvas id="overlay"></canvas>
        </div>
        <div id="stream_status" class="muted" style="margin-top:8px;">Stream verbindet...</div>
        <div class="muted" style="margin-top:8px;">Klicke im Feed, um Zonenpunkte zu setzen. In `dual_polygon` bearbeitest du Entry/Exit getrennt.</div>
      </div>

      <div class="panel">
        <h3 class="title">Controls</h3>
        <div class="row">
          <div class="stat"><b>Display Occ</b><div id="occupancy">-</div></div>
          <div class="stat"><b>Video Occ</b><div id="video_occupancy">-</div></div>
          <div class="stat"><b>Sim Occ</b><div id="sim_occupancy">-</div></div>
          <div class="stat"><b>Entries</b><div id="entries">-</div></div>
          <div class="stat"><b>Exits</b><div id="exits">-</div></div>
        </div>
        <div class="row">
          <div class="stat"><b>Tracks</b><div id="tracks">-</div></div>
          <div class="stat"><b>FPS</b><div id="fps">-</div></div>
          <div class="stat"><b>Infer FPS</b><div id="infer_fps">-</div></div>
          <div class="stat"><b>Detector</b><div id="detector_source">-</div></div>
          <div class="stat"><b>Frame Ev.</b><div id="events">-</div></div>
        </div>
        <div id="status" class="muted">Loading...</div>
        <div class="row" style="margin-top:10px; align-items:center;">
          <button id="toggle_analysis">Auswertung stoppen</button>
          <div class="muted">Status: <span id="analysis_status" class="ok">aktiv</span></div>
        </div>
        <div class="row" style="margin-top:10px; align-items:center; gap:6px;">
          <label style="margin-right:4px;">Detektor:</label>
          <button id="det_local" class="det-btn">Local</button>
          <button id="det_api" class="det-btn">API</button>
          <button id="det_hybrid" class="det-btn">Hybrid</button>
          <span id="det_status" class="muted"></span>
        </div>
        <div class="row" style="margin-top:10px;">
          <a class="navlink" id="main_link">Zur Hauptseite</a>
          <a class="navlink" id="analytics_link" target="_blank" rel="noopener">Analytics öffnen</a>
          <a class="navlink" id="api_link" target="_blank" rel="noopener">API/Command Center</a>
        </div>

        <h3 class="title" style="margin-top:16px;">Video Source</h3>
        <div class="row">
          <label>Mode</label>
          <select id="video_mode">
            <option value="youtube">youtube</option>
            <option value="livefeed_simulation">Livefeed Simulation</option>
          </select>
        </div>
        <div class="row">
          <label>Source URL/Pfad</label>
          <input id="video_source" type="text" placeholder="https://youtube.com/watch?v=..." style="flex: 1 1 360px;" />
        </div>
        <div class="row">
          <button id="save_source">Save Source</button>
          <button id="reload_source">Reload Source</button>
        </div>
        <div id="video_source_status" class="muted">Source wird geladen...</div>

        <h3 class="title" style="margin-top:16px;">Simulation Queue</h3>
        <div class="sim-queue">
          <div class="row">
            <label>Control</label>
            <select id="sim_control_mode">
              <option value="remote_control">remote_control</option>
              <option value="auto_rules">auto_rules</option>
            </select>
            <label><input id="sim_idle_loop" type="checkbox"> Leerlauf loopen</label>
            <button id="sim_reload_catalog">Reload Clips</button>
          </div>
          <div class="row">
            <label>Clip</label>
            <select id="sim_clip_select" class="sim-select"></select>
          </div>
          <div class="row">
            <button id="sim_play_now">Play Now</button>
            <button id="sim_play_next">Play Next</button>
            <button id="sim_enqueue">Enqueue</button>
            <button id="sim_clear_queue">Queue leeren</button>
          </div>
          <div id="sim_catalog_info" class="muted">Catalog lädt...</div>
          <div id="sim_active_info" class="muted">Aktiver Clip: -</div>
          <div id="sim_queue_board" class="queue-board">
            <span class="queue-pill basic">Queue leer</span>
          </div>
        </div>

        <h3 class="title" style="margin-top:16px;">Jahresprofil-Simulation</h3>
        <div class="sim-queue">
          <div class="row">
            <label><input id="profile_sim_enabled" type="checkbox"> Profilroutine aktiv</label>
            <button id="profile_sim_reload">Laden</button>
            <button id="profile_sim_save">Speichern</button>
          </div>
          <div class="row">
            <label>Excel-Pfad</label>
            <input id="profile_sim_excel" type="text" style="flex: 1 1 340px;" placeholder="KI_Projekt_Daten_einJahr.xlsx" />
          </div>
          <div class="row">
            <label>Tick (s)</label>
            <input id="profile_sim_tick" type="number" min="60" max="600" step="1" value="60" style="width:95px;" />
            <label>Rollback (min)</label>
            <input id="profile_sim_rollback" type="number" min="1" max="240" step="1" value="15" style="width:95px;" />
          </div>
          <div class="row">
            <label>Profil-Bindung</label>
            <input id="profile_sim_blend" type="number" min="0.05" max="1" step="0.01" value="0.72" style="width:95px;" />
            <label>Rauschfaktor</label>
            <input id="profile_sim_noise" type="number" min="0" max="4" step="0.05" value="0.85" style="width:95px;" />
            <label>Max Schritt/Tick</label>
            <input id="profile_sim_step" type="number" min="0.2" max="20" step="0.1" value="2" style="width:95px;" />
          </div>
          <div id="profile_sim_info" class="muted">Profilroutine lädt...</div>
        </div>

        <div class="row" style="margin-top:10px;">
          <label>Mode</label>
          <select id="mode">
            <option value="line">line</option>
            <option value="polygon">polygon</option>
            <option value="dual_polygon">dual_polygon</option>
          </select>
          <label>Direction</label>
          <select id="direction">
            <option value="negative_to_positive">negative_to_positive</option>
            <option value="positive_to_negative">positive_to_negative</option>
          </select>
        </div>

        <div class="row">
          <label>Active Polygon</label>
          <select id="activePoly">
            <option value="entry">entry</option>
            <option value="exit">exit</option>
          </select>
        </div>

        <div class="row">
          <button id="undo">Undo Polygon Point</button>
          <button id="clear">Clear Polygon</button>
        </div>
        <div class="row">
          <button id="save">Save Zone</button>
          <button id="reload">Reload Zone</button>
        </div>
        <div class="muted">Current line clicks: <span id="line_stage">0</span>/2</div>

        <h3 class="title" style="margin-top:16px;">Analysis ROI</h3>
        <div class="row">
          <label><input type="checkbox" id="roi_enabled"> Enable Crop</label>
          <label>Mode</label>
          <select id="roi_mode">
            <option value="rect">rect</option>
            <option value="polygon">polygon</option>
          </select>
          <label><input type="checkbox" id="roi_edit_mode"> ROI Edit</label>
        </div>
        <div class="row">
          <label>X min</label><input id="roi_xmin" type="number" min="0" max="1" step="0.01" value="0" style="width:90px;" />
          <label>Y min</label><input id="roi_ymin" type="number" min="0" max="1" step="0.01" value="0" style="width:90px;" />
        </div>
        <div class="row">
          <label>X max</label><input id="roi_xmax" type="number" min="0" max="1" step="0.01" value="1" style="width:90px;" />
          <label>Y max</label><input id="roi_ymax" type="number" min="0" max="1" step="0.01" value="1" style="width:90px;" />
        </div>
        <div class="row">
          <button id="save_roi">Save ROI</button>
          <button id="reload_roi">Reload ROI</button>
        </div>
        <div class="row">
          <button id="undo_roi">Undo ROI Point</button>
          <button id="clear_roi">Clear ROI Polygon</button>
        </div>
      </div>
    </div>
  </div>

  <script src="https://cdn.dashjs.org/latest/dash.all.min.js"></script>
  <script>
    const dashFeed = document.getElementById('dashFeed');
    const canvas = document.getElementById('overlay');
    const ctx = canvas.getContext('2d');

    const modeSel = document.getElementById('mode');
    const directionSel = document.getElementById('direction');
    const lineStageEl = document.getElementById('line_stage');

    const occupancyEl = document.getElementById('occupancy');
    const videoOccupancyEl = document.getElementById('video_occupancy');
    const simOccupancyEl = document.getElementById('sim_occupancy');
    const entriesEl = document.getElementById('entries');
    const exitsEl = document.getElementById('exits');
    const tracksEl = document.getElementById('tracks');
    const fpsEl = document.getElementById('fps');
    const inferFpsEl = document.getElementById('infer_fps');
    const detectorSourceEl = document.getElementById('detector_source');
    const eventsEl = document.getElementById('events');
    const statusEl = document.getElementById('status');
    const toggleAnalysisBtn = document.getElementById('toggle_analysis');
    const analysisStatusEl = document.getElementById('analysis_status');
    const streamStatusEl = document.getElementById('stream_status');
    const mainLinkEl = document.getElementById('main_link');
    const analyticsLinkEl = document.getElementById('analytics_link');
    const apiLinkEl = document.getElementById('api_link');
    const videoModeEl = document.getElementById('video_mode');
    const videoSourceEl = document.getElementById('video_source');
    const videoSourceStatusEl = document.getElementById('video_source_status');
    const simControlModeEl = document.getElementById('sim_control_mode');
    const simIdleLoopEl = document.getElementById('sim_idle_loop');
    const simClipSelectEl = document.getElementById('sim_clip_select');
    const simCatalogInfoEl = document.getElementById('sim_catalog_info');
    const simActiveInfoEl = document.getElementById('sim_active_info');
    const simQueueBoardEl = document.getElementById('sim_queue_board');
    const profileSimEnabledEl = document.getElementById('profile_sim_enabled');
    const profileSimExcelEl = document.getElementById('profile_sim_excel');
    const profileSimTickEl = document.getElementById('profile_sim_tick');
    const profileSimRollbackEl = document.getElementById('profile_sim_rollback');
    const profileSimBlendEl = document.getElementById('profile_sim_blend');
    const profileSimNoiseEl = document.getElementById('profile_sim_noise');
    const profileSimStepEl = document.getElementById('profile_sim_step');
    const profileSimInfoEl = document.getElementById('profile_sim_info');
    const activePolySel = document.getElementById('activePoly');
    const roiEnabledEl = document.getElementById('roi_enabled');
    const roiModeEl = document.getElementById('roi_mode');
    const roiEditModeEl = document.getElementById('roi_edit_mode');
    const roiXMinEl = document.getElementById('roi_xmin');
    const roiYMinEl = document.getElementById('roi_ymin');
    const roiXMaxEl = document.getElementById('roi_xmax');
    const roiYMaxEl = document.getElementById('roi_ymax');
    const DASH_ENABLED = {{ dash_enabled|tojson }};
    const HOST = window.location.hostname || '127.0.0.1';
    const ACCESS_HOST = HOST === '0.0.0.0' ? '127.0.0.1' : HOST;
    const PATHNAME = window.location.pathname || '/';
    const IN_PORTAL_PREFIX = PATHNAME === '/realtime' || PATHNAME.startsWith('/realtime/');
    const URL_PREFIX = IN_PORTAL_PREFIX ? '/realtime' : '';
    const withPrefix = (path) => `${URL_PREFIX}${path}`;

    let zone = {
      mode: 'line',
      line: { p1: [0.35, 0.65], p2: [0.65, 0.65], entry_direction: 'negative_to_positive' },
      polygon: { points: [] },
      entry_polygon: { points: [] },
      exit_polygon: { points: [] },
      min_crossing_displacement_px: 20.0,
      min_track_points: 4,
      min_event_cooldown_frames: 8
    };
    let lineStage = 0;
    let analysisRoi = { enabled: false, mode: 'rect', x_min: 0.0, y_min: 0.0, x_max: 1.0, y_max: 1.0, polygon_points: [] };
    let analysisEnabled = true;
    let dashController = null;
    let simulationCatalog = [];

    function setAnalysisUi(enabled) {
      analysisEnabled = !!enabled;
      toggleAnalysisBtn.textContent = analysisEnabled ? 'Auswertung stoppen' : 'Auswertung starten';
      analysisStatusEl.textContent = analysisEnabled ? 'aktiv' : 'pausiert';
      analysisStatusEl.className = analysisEnabled ? 'ok' : 'warn';
    }

    function clamp01(value) {
      return Math.max(0, Math.min(1, value));
    }

    function initDashPlayer() {
      if (!DASH_ENABLED) {
        streamStatusEl.textContent = 'DASH ist deaktiviert';
        return;
      }
      if (!window.dashjs) {
        streamStatusEl.textContent = 'dash.js konnte nicht geladen werden';
        return;
      }

      streamStatusEl.textContent = 'DASH wird initialisiert...';
      const dashUrl = withPrefix(`/dash/stream.mpd?_t=${Date.now()}`);
      dashController = window.dashjs.MediaPlayer().create();
      dashController.updateSettings({
        streaming: {
          lowLatencyEnabled: true,
          liveDelay: 2,
        },
      });
      dashController.initialize(dashFeed, dashUrl, true);
      dashController.on(window.dashjs.MediaPlayer.events.STREAM_INITIALIZED, () => {
        streamStatusEl.textContent = 'DASH aktiv';
        dashFeed.playbackRate = 1.0;
        dashFeed.play().catch(() => {});
      });
      dashController.on(window.dashjs.MediaPlayer.events.ERROR, () => {
        streamStatusEl.textContent = 'DASH Fehler: Stream neu laden';
      });
    }

    function syncCanvasSize() {
      const rect = dashFeed.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width));
      canvas.height = Math.max(1, Math.floor(rect.height));
    }

    function getVideoGeometry() {
      const canvasW = Math.max(1, canvas.width);
      const canvasH = Math.max(1, canvas.height);
      const videoW = Math.max(1, dashFeed.videoWidth || canvasW);
      const videoH = Math.max(1, dashFeed.videoHeight || canvasH);
      const scale = Math.min(canvasW / videoW, canvasH / videoH);
      const drawW = Math.max(1, videoW * scale);
      const drawH = Math.max(1, videoH * scale);
      const offsetX = (canvasW - drawW) * 0.5;
      const offsetY = (canvasH - drawH) * 0.5;
      return { canvasW, canvasH, drawW, drawH, offsetX, offsetY };
    }

    function normToCanvas(nx, ny, geom) {
      return [
        geom.offsetX + (clamp01(nx) * geom.drawW),
        geom.offsetY + (clamp01(ny) * geom.drawH),
      ];
    }

    function drawZone() {
      syncCanvasSize();
      const geom = getVideoGeometry();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.lineWidth = 2;

      if (zone.mode === 'line') {
        const p1 = normToCanvas(zone.line.p1[0], zone.line.p1[1], geom);
        const p2 = normToCanvas(zone.line.p2[0], zone.line.p2[1], geom);
        ctx.strokeStyle = '#00ffff';
        ctx.beginPath();
        ctx.moveTo(p1[0], p1[1]);
        ctx.lineTo(p2[0], p2[1]);
        ctx.stroke();
        ctx.fillStyle = '#ffff00';
        ctx.beginPath(); ctx.arc(p1[0], p1[1], 5, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(p2[0], p2[1], 5, 0, Math.PI * 2); ctx.fill();
      } else if (zone.mode === 'polygon') {
        const pts = zone.polygon.points.map(p => normToCanvas(p[0], p[1], geom));
        if (pts.length >= 2) {
          ctx.strokeStyle = '#ff7f50';
          ctx.beginPath();
          ctx.moveTo(pts[0][0], pts[0][1]);
          for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
          if (pts.length >= 3) ctx.closePath();
          ctx.stroke();
        }
        ctx.fillStyle = '#ff7f50';
        for (const p of pts) { ctx.beginPath(); ctx.arc(p[0], p[1], 4, 0, Math.PI * 2); ctx.fill(); }
      } else {
        const entryPts = zone.entry_polygon.points.map(p => normToCanvas(p[0], p[1], geom));
        const exitPts = zone.exit_polygon.points.map(p => normToCanvas(p[0], p[1], geom));

        if (entryPts.length >= 2) {
          ctx.strokeStyle = '#22c55e';
          ctx.beginPath();
          ctx.moveTo(entryPts[0][0], entryPts[0][1]);
          for (let i = 1; i < entryPts.length; i++) ctx.lineTo(entryPts[i][0], entryPts[i][1]);
          if (entryPts.length >= 3) ctx.closePath();
          ctx.stroke();
        }
        ctx.fillStyle = '#22c55e';
        for (const p of entryPts) { ctx.beginPath(); ctx.arc(p[0], p[1], 4, 0, Math.PI * 2); ctx.fill(); }

        if (exitPts.length >= 2) {
          ctx.strokeStyle = '#ef4444';
          ctx.beginPath();
          ctx.moveTo(exitPts[0][0], exitPts[0][1]);
          for (let i = 1; i < exitPts.length; i++) ctx.lineTo(exitPts[i][0], exitPts[i][1]);
          if (exitPts.length >= 3) ctx.closePath();
          ctx.stroke();
        }
        ctx.fillStyle = '#ef4444';
        for (const p of exitPts) { ctx.beginPath(); ctx.arc(p[0], p[1], 4, 0, Math.PI * 2); ctx.fill(); }
      }

      if (analysisRoi && analysisRoi.enabled) {
        ctx.strokeStyle = '#ff00ff';
        ctx.lineWidth = 2;
        if ((analysisRoi.mode || 'rect') === 'polygon') {
          const pts = (analysisRoi.polygon_points || []).map(p => normToCanvas(p[0], p[1], geom));
          if (pts.length >= 2) {
            ctx.beginPath();
            ctx.moveTo(pts[0][0], pts[0][1]);
            for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
            if (pts.length >= 3) ctx.closePath();
            ctx.stroke();
          }
          ctx.fillStyle = '#ff00ff';
          for (const p of pts) { ctx.beginPath(); ctx.arc(p[0], p[1], 4, 0, Math.PI * 2); ctx.fill(); }
        } else {
          const [rx1, ry1] = normToCanvas(analysisRoi.x_min, analysisRoi.y_min, geom);
          const [rx2, ry2] = normToCanvas(analysisRoi.x_max, analysisRoi.y_max, geom);
          ctx.strokeRect(rx1, ry1, rx2 - rx1, ry2 - ry1);
        }
      }
      lineStageEl.textContent = String(lineStage);
    }

    function setRoiForm(roi) {
      analysisRoi = roi || analysisRoi;
      roiEnabledEl.checked = !!analysisRoi.enabled;
      roiModeEl.value = String(analysisRoi.mode || 'rect');
      roiXMinEl.value = Number(analysisRoi.x_min ?? 0).toFixed(2);
      roiYMinEl.value = Number(analysisRoi.y_min ?? 0).toFixed(2);
      roiXMaxEl.value = Number(analysisRoi.x_max ?? 1).toFixed(2);
      roiYMaxEl.value = Number(analysisRoi.y_max ?? 1).toFixed(2);
      if (!Array.isArray(analysisRoi.polygon_points)) analysisRoi.polygon_points = [];
    }

    function getRoiFromForm() {
      const clip = (v) => Math.max(0, Math.min(1, Number(v) || 0));
      let xMin = clip(roiXMinEl.value);
      let yMin = clip(roiYMinEl.value);
      let xMax = clip(roiXMaxEl.value);
      let yMax = clip(roiYMaxEl.value);

      if (xMax <= xMin) xMax = Math.min(1, xMin + 0.01);
      if (yMax <= yMin) yMax = Math.min(1, yMin + 0.01);

      return {
        enabled: !!roiEnabledEl.checked,
        mode: roiModeEl.value === 'polygon' ? 'polygon' : 'rect',
        x_min: xMin,
        y_min: yMin,
        x_max: xMax,
        y_max: yMax,
        polygon_points: Array.isArray(analysisRoi.polygon_points) ? analysisRoi.polygon_points : [],
      };
    }

    async function loadRoi() {
      const res = await fetch(withPrefix('/api/tracking-roi'));
      const payload = await res.json();
      setRoiForm(payload.analysis_roi || payload);
      drawZone();
    }

    async function saveRoi() {
      const roi = getRoiFromForm();
      const res = await fetch(withPrefix('/api/tracking-roi'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ analysis_roi: roi }),
      });
      const data = await res.json();
      if (data.ok) {
        setRoiForm(data.analysis_roi || roi);
        statusEl.textContent = 'ROI saved and applied.';
        statusEl.className = 'ok';
      } else {
        statusEl.textContent = 'ROI error: ' + (data.error || 'unknown');
        statusEl.className = 'warn';
      }
      drawZone();
    }

    function clickToNorm(ev) {
      const rect = canvas.getBoundingClientRect();
      const geom = getVideoGeometry();
      const xCanvas = ((ev.clientX - rect.left) / Math.max(1, rect.width)) * canvas.width;
      const yCanvas = ((ev.clientY - rect.top) / Math.max(1, rect.height)) * canvas.height;
      const nx = (xCanvas - geom.offsetX) / Math.max(1, geom.drawW);
      const ny = (yCanvas - geom.offsetY) / Math.max(1, geom.drawH);
      return [clamp01(nx), clamp01(ny)];
    }

    function hasSimulationOption(clipId) {
      const normalized = String(clipId || '').trim();
      if (!normalized) return false;
      return Array.from(simClipSelectEl.options).some((opt) => opt.value === normalized);
    }

    async function loadVideoSource(preferActiveSelection = false) {
      const res = await fetch(withPrefix('/api/video-source'));
      const payload = await res.json();
      const mode = String(payload.mode || 'youtube');
      videoModeEl.value = mode === 'livefeed_simulation' ? 'livefeed_simulation' : 'youtube';
      videoSourceEl.value = String(payload.source || '');
      const active = String(payload.active_source || '');
      videoSourceStatusEl.textContent = active ? `Aktiv: ${active}` : 'Keine aktive Source';
      updateSimulationPanelFromPayload(payload, preferActiveSelection);
    }

    function renderSimulationQueue(payload) {
      const active = payload?.simulation_active_clip || {};
      const basic = payload?.simulation_basic_clip || {};
      const queue = Array.isArray(payload?.simulation_pending_queue) ? payload.simulation_pending_queue : [];
      const activeKind = String(payload?.simulation_active_kind || 'basic');
      const idleLoop = !!payload?.simulation_idle_loop;

      const html = [];
      if (active.clip_id) {
        html.push(`<span class="queue-pill active">NOW: ${active.display_name || active.clip_id} (${activeKind})</span>`);
      }
      for (const item of queue) {
        html.push(`<span class="queue-pill">NEXT: ${item.display_name || item.clip_id}</span>`);
      }
      if (basic.clip_id) {
        const basicMode = idleLoop ? 'LOOP' : 'HOLD';
        html.push(`<span class="queue-pill basic">${basicMode}: ${basic.display_name || basic.clip_id}</span>`);
      }

      simQueueBoardEl.innerHTML = html.length ? html.join('') : '<span class="queue-pill basic">Queue leer</span>';
    }

    function updateSimulationPanelFromPayload(payload, forceClipSelection = false) {
      const controlMode = String(payload?.simulation_control_mode || 'remote_control');
      simControlModeEl.value = controlMode === 'auto_rules' ? 'auto_rules' : 'remote_control';
      const idleLoop = !!payload?.simulation_idle_loop;
      simIdleLoopEl.checked = idleLoop;

      const active = payload?.simulation_active_clip || {};
      const basic = payload?.simulation_basic_clip || {};
      const pendingCount = Number(payload?.simulation_pending_count || 0);
      const activeLabel = active.display_name || active.clip_id || '-';
      const basicLabel = basic.display_name || basic.clip_id || '-';
      simActiveInfoEl.textContent = `Aktiv: ${activeLabel} | Basic: ${basicLabel} | Queue: ${pendingCount} | Idle: ${idleLoop ? 'Loop' : 'Standbild'}`;

      const selectedId = String(payload?.simulation_clip_id || payload?.simulation_active_clip_id || '');
      const currentSelection = String(simClipSelectEl.value || '').trim();
      const shouldSyncSelection = forceClipSelection || !hasSimulationOption(currentSelection);
      if (shouldSyncSelection && hasSimulationOption(selectedId)) {
        simClipSelectEl.value = selectedId;
      }

      renderSimulationQueue(payload);
    }

    function updateProfileSimulationPanelFromPayload(payload) {
      const sim = payload?.profile_simulation || {};
      profileSimEnabledEl.checked = !!sim.enabled;
      profileSimExcelEl.value = String(sim.excel_path || profileSimExcelEl.value || '');
      profileSimTickEl.value = Number(sim.tick_seconds ?? 60).toFixed(0);
      profileSimRollbackEl.value = Number(sim.rollback_minutes ?? 15).toFixed(0);
      profileSimBlendEl.value = Number(sim.profile_blend ?? 0.72).toFixed(2);
      profileSimNoiseEl.value = Number(sim.noise_sigma_scale ?? 0.85).toFixed(2);
      profileSimStepEl.value = Number(sim.max_step_per_tick ?? 2).toFixed(1);

      const occ = Number(sim.simulated_occupancy ?? 0);
      const loaded = !!sim.profile_loaded;
      const err = String(sim.profile_error || '').trim();
      const buckets = Number(sim.profile_bucket_count || 0);
      const simEntriesTotal = Number(sim.sim_entries_total || 0);
      const simExitsTotal = Number(sim.sim_exits_total || 0);
      const pauseHint = (!analysisEnabled && !!sim.enabled)
        ? ' | aktiv waehrend Pause'
        : '';
      if (err) {
        profileSimInfoEl.textContent = `Status: Fehler (${err}) | Occ: ${occ} | Sim +${simEntriesTotal} / -${simExitsTotal}${pauseHint}`;
      } else {
        profileSimInfoEl.textContent = `Status: ${loaded ? 'Profil geladen' : 'Profil ausstehend'} | Buckets: ${buckets} | Sim Occ: ${occ} | Sim +${simEntriesTotal} / -${simExitsTotal}${pauseHint}`;
      }
    }

    async function loadProfileSimulationControl() {
      const res = await fetch(withPrefix('/api/profile-simulation'));
      const payload = await res.json();
      if (!payload.ok) {
        profileSimInfoEl.textContent = `Laden fehlgeschlagen: ${payload.error || 'unknown'}`;
        return;
      }
      updateProfileSimulationPanelFromPayload(payload);
    }

    async function saveProfileSimulationControl() {
      const profileSimulation = {
        enabled: !!profileSimEnabledEl.checked,
        excel_path: String(profileSimExcelEl.value || '').trim(),
        tick_seconds: Number(profileSimTickEl.value || 60),
        rollback_minutes: Number(profileSimRollbackEl.value || 15),
        profile_blend: Number(profileSimBlendEl.value || 0.72),
        noise_sigma_scale: Number(profileSimNoiseEl.value || 0.85),
        max_step_per_tick: Number(profileSimStepEl.value || 2),
      };

      const res = await fetch(withPrefix('/api/profile-simulation'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ profile_simulation: profileSimulation }),
      });
      const data = await res.json();
      if (!data.ok) {
        statusEl.textContent = `Profilsimulation Fehler: ${data.error || 'unknown'}`;
        statusEl.className = 'warn';
        return;
      }
      updateProfileSimulationPanelFromPayload(data);
      statusEl.textContent = 'Profilsimulation aktualisiert.';
      statusEl.className = 'ok';
    }

    async function loadSimulationCatalog() {
      const res = await fetch(withPrefix('/api/simulation/catalog'));
      const payload = await res.json();
      if (!payload.ok) {
        simCatalogInfoEl.textContent = `Catalog Fehler: ${payload.error || 'unknown'}`;
        return;
      }

      const previousSelection = String(simClipSelectEl.value || '').trim();
      simulationCatalog = Array.isArray(payload.clips) ? payload.clips : [];
      simClipSelectEl.innerHTML = '';
      for (const clip of simulationCatalog) {
        const option = document.createElement('option');
        option.value = String(clip.clip_id || '');
        const displayPath = String(clip.display_relative_path || clip.relative_path || '');
        const aliases = Number(clip.aliases_count || 0);
        option.textContent = aliases > 0
          ? `${clip.display_name || clip.clip_id} (${displayPath}) [dup:${aliases}]`
          : `${clip.display_name || clip.clip_id} (${displayPath})`;
        simClipSelectEl.appendChild(option);
      }

      if (hasSimulationOption(previousSelection)) {
        simClipSelectEl.value = previousSelection;
      }

      const summary = payload.summary || {};
      simCatalogInfoEl.textContent = `Clips: ${summary.clips || simulationCatalog.length || 0} | Duplikate: ${summary.duplicates || 0}`;

      await loadVideoSource(true);
    }

    async function simulationAction(action) {
      const payload = {
        action,
        control_mode: simControlModeEl.value === 'auto_rules' ? 'auto_rules' : 'remote_control',
        idle_loop: !!simIdleLoopEl.checked,
      };
      if (action !== 'clear_queue') {
        const clipId = String(simClipSelectEl.value || '').trim();
        if (!clipId) {
          statusEl.textContent = 'Kein Clip ausgewählt.';
          statusEl.className = 'warn';
          return;
        }
        payload.clip_id = clipId;
      }

      const res = await fetch(withPrefix('/api/simulation/select'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!data.ok) {
        statusEl.textContent = `Simulation Fehler: ${data.error || 'unknown'}`;
        statusEl.className = 'warn';
        return;
      }

      const selected = data.selected?.display_name || data.selected?.clip_id || '-';
      if (action === 'enqueue') {
        statusEl.textContent = `In Queue: ${selected}`;
      } else if (action === 'play_next') {
        statusEl.textContent = `Als Nächstes: ${selected}`;
      } else if (action === 'clear_queue') {
        statusEl.textContent = 'Queue geleert.';
      } else {
        statusEl.textContent = `Spiele jetzt: ${selected}`;
      }
      statusEl.className = 'ok';

      await loadVideoSource();
    }

    async function saveSimulationControl() {
      const payload = {
        control_mode: simControlModeEl.value === 'auto_rules' ? 'auto_rules' : 'remote_control',
        idle_loop: !!simIdleLoopEl.checked,
      };

      const res = await fetch(withPrefix('/api/simulation/control'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!data.ok) {
        statusEl.textContent = `Simulation-Control Fehler: ${data.error || 'unknown'}`;
        statusEl.className = 'warn';
        return;
      }

      updateSimulationPanelFromPayload(data.video_source || data);
      statusEl.textContent = 'Simulation-Control gespeichert.';
      statusEl.className = 'ok';
    }

    async function saveVideoSource() {
      const mode = videoModeEl.value === 'livefeed_simulation' ? 'livefeed_simulation' : 'youtube';
      const source = String(videoSourceEl.value || '').trim();
      const res = await fetch(withPrefix('/api/video-source'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode,
          source,
          simulation_control_mode: simControlModeEl.value === 'auto_rules' ? 'auto_rules' : 'remote_control',
          simulation_idle_loop: !!simIdleLoopEl.checked,
        }),
      });
      const data = await res.json();
      if (data.ok) {
        const openedTag = data.opened ? 'ok' : 'pending';
        videoSourceStatusEl.textContent = `Source gespeichert (${openedTag}) | aktiv: ${data.active_source || '-'}`;
        statusEl.textContent = 'Video source updated.';
        statusEl.className = 'ok';
      } else {
        videoSourceStatusEl.textContent = 'Source update fehlgeschlagen';
        statusEl.textContent = 'Source error: ' + (data.error || 'unknown');
        statusEl.className = 'warn';
      }
    }

    async function saveZone() {
      zone.mode = modeSel.value;
      zone.line.entry_direction = directionSel.value;
      const res = await fetch(withPrefix('/api/zone'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(zone)
      });
      const data = await res.json();
      statusEl.textContent = data.ok ? 'Zone saved and applied.' : ('Error: ' + (data.error || 'unknown'));
      statusEl.className = data.ok ? 'ok' : 'warn';
    }

    async function reloadZone() {
      const res = await fetch(withPrefix('/api/zone'));
      zone = await res.json();
      modeSel.value = zone.mode;
      directionSel.value = zone.line.entry_direction;
      lineStage = 0;
      drawZone();
    }

    async function pollState() {
      try {
        const res = await fetch(withPrefix('/api/state'));
        const st = await res.json();
        occupancyEl.textContent = st.occupancy;
        videoOccupancyEl.textContent = st.video_occupancy ?? st.frame_occupancy ?? 0;
        simOccupancyEl.textContent = st.simulated_occupancy ?? 0;
        entriesEl.textContent = st.entries_total;
        exitsEl.textContent = st.exits_total;
        tracksEl.textContent = st.tracks;
        fpsEl.textContent = st.fps;
        inferFpsEl.textContent = st.inference_fps;
        detectorSourceEl.textContent = st.detector_source ?? '-';
        eventsEl.textContent = `+${st.entries_frame} / -${st.exits_frame}`;
        if (st.detector_mode) updateDetectorButtons(st.detector_mode);
        setAnalysisUi(st.analysis_enabled !== false);
        updateSimulationPanelFromPayload(st);
        updateProfileSimulationPanelFromPayload(st);
      } catch (e) {}
    }

    async function toggleAnalysis() {
      toggleAnalysisBtn.disabled = true;
      try {
        const res = await fetch(withPrefix('/api/analysis-control'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ enabled: !analysisEnabled }),
        });
        const data = await res.json();
        if (data.ok) {
          setAnalysisUi(data.analysis_enabled);
          statusEl.textContent = data.analysis_enabled
            ? 'Auswertung gestartet.'
            : 'Auswertung pausiert. Keine Inferenz-API-Aufrufe aktiv.';
          statusEl.className = 'ok';
        } else {
          statusEl.textContent = 'Steuerung fehlgeschlagen: ' + (data.error || 'unknown');
          statusEl.className = 'warn';
        }
      } catch (e) {
        statusEl.textContent = 'Steuerung fehlgeschlagen';
        statusEl.className = 'warn';
      } finally {
        toggleAnalysisBtn.disabled = false;
      }
    }

    canvas.addEventListener('click', (ev) => {
      const [nx, ny] = clickToNorm(ev);

      if (roiEditModeEl.checked) {
        const mode = roiModeEl.value;
        if (mode === 'polygon') {
          if (!Array.isArray(analysisRoi.polygon_points)) analysisRoi.polygon_points = [];
          analysisRoi.polygon_points.push([nx, ny]);
        } else {
          const xMin = Number(roiXMinEl.value);
          const yMin = Number(roiYMinEl.value);
          const xMax = Number(roiXMaxEl.value);
          const yMax = Number(roiYMaxEl.value);
          const hasRect = xMax > xMin && yMax > yMin;
          if (!hasRect || !analysisRoi._rectStage) {
            roiXMinEl.value = nx.toFixed(2);
            roiYMinEl.value = ny.toFixed(2);
            roiXMaxEl.value = nx.toFixed(2);
            roiYMaxEl.value = ny.toFixed(2);
            analysisRoi._rectStage = 1;
          } else {
            roiXMaxEl.value = nx.toFixed(2);
            roiYMaxEl.value = ny.toFixed(2);
            analysisRoi._rectStage = 0;
          }
        }
        analysisRoi = getRoiFromForm();
        drawZone();
        return;
      }

      zone.mode = modeSel.value;
      if (zone.mode === 'line') {
        if (lineStage === 0) {
          zone.line.p1 = [nx, ny];
          lineStage = 1;
        } else {
          zone.line.p2 = [nx, ny];
          lineStage = 0;
        }
      } else if (zone.mode === 'polygon') {
        zone.polygon.points.push([nx, ny]);
      } else {
        const key = activePolySel.value === 'exit' ? 'exit_polygon' : 'entry_polygon';
        zone[key].points.push([nx, ny]);
      }
      drawZone();
    });

    document.getElementById('undo').addEventListener('click', () => {
      if (zone.mode === 'polygon') {
        if (zone.polygon.points.length) zone.polygon.points.pop();
      } else if (zone.mode === 'dual_polygon') {
        const key = activePolySel.value === 'exit' ? 'exit_polygon' : 'entry_polygon';
        if (zone[key].points.length) zone[key].points.pop();
      }
      drawZone();
    });

    document.getElementById('clear').addEventListener('click', () => {
      if (zone.mode === 'polygon') {
        zone.polygon.points = [];
      } else if (zone.mode === 'dual_polygon') {
        const key = activePolySel.value === 'exit' ? 'exit_polygon' : 'entry_polygon';
        zone[key].points = [];
      }
      drawZone();
    });

    document.getElementById('save').addEventListener('click', saveZone);
    document.getElementById('reload').addEventListener('click', reloadZone);
    toggleAnalysisBtn.addEventListener('click', toggleAnalysis);
    document.getElementById('save_source').addEventListener('click', saveVideoSource);
    document.getElementById('reload_source').addEventListener('click', loadVideoSource);
    document.getElementById('sim_reload_catalog').addEventListener('click', loadSimulationCatalog);
    document.getElementById('sim_play_now').addEventListener('click', () => simulationAction('play_now'));
    document.getElementById('sim_play_next').addEventListener('click', () => simulationAction('play_next'));
    document.getElementById('sim_enqueue').addEventListener('click', () => simulationAction('enqueue'));
    document.getElementById('sim_clear_queue').addEventListener('click', () => simulationAction('clear_queue'));
    simControlModeEl.addEventListener('change', saveSimulationControl);
    simIdleLoopEl.addEventListener('change', saveSimulationControl);
    document.getElementById('profile_sim_reload').addEventListener('click', loadProfileSimulationControl);
    document.getElementById('profile_sim_save').addEventListener('click', saveProfileSimulationControl);
    profileSimEnabledEl.addEventListener('change', saveProfileSimulationControl);
    document.getElementById('save_roi').addEventListener('click', saveRoi);
    document.getElementById('reload_roi').addEventListener('click', loadRoi);
    document.getElementById('undo_roi').addEventListener('click', () => {
      if (!Array.isArray(analysisRoi.polygon_points)) analysisRoi.polygon_points = [];
      if (analysisRoi.polygon_points.length) analysisRoi.polygon_points.pop();
      drawZone();
    });
    document.getElementById('clear_roi').addEventListener('click', () => {
      analysisRoi.polygon_points = [];
      drawZone();
    });
    modeSel.addEventListener('change', drawZone);
    directionSel.addEventListener('change', drawZone);
    activePolySel.addEventListener('change', drawZone);
    roiModeEl.addEventListener('change', () => {
      analysisRoi = getRoiFromForm();
      drawZone();
    });

    window.addEventListener('resize', drawZone);
    dashFeed.addEventListener('ratechange', () => {
      if (dashFeed.playbackRate !== 1.0) dashFeed.playbackRate = 1.0;
    });

    if (IN_PORTAL_PREFIX) {
      mainLinkEl.href = '/';
      analyticsLinkEl.href = '/analytics';
      apiLinkEl.href = '/api/v1/dashboard/command-center?zone_id=default-zone&horizon=210&history_minutes=180';
    } else {
      mainLinkEl.href = `http://${ACCESS_HOST}:8090`;
      analyticsLinkEl.href = `http://${ACCESS_HOST}:8501`;
      apiLinkEl.href = `http://${ACCESS_HOST}:8000/api/v1/dashboard/command-center?zone_id=default-zone&horizon=210&history_minutes=180`;
    }

    reloadZone();
    loadRoi();
    loadVideoSource();
    loadSimulationCatalog();
    loadProfileSimulationControl();
    initDashPlayer();

    function updateDetectorButtons(mode) {
      document.querySelectorAll('.det-btn').forEach(b => b.classList.remove('active'));
      const btn = document.getElementById('det_' + mode);
      if (btn) btn.classList.add('active');
    }
    async function setDetectorMode(mode) {
      document.querySelectorAll('.det-btn').forEach(b => b.disabled = true);
      try {
        const res = await fetch(withPrefix('/api/detector/mode'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ mode }),
        });
        const data = await res.json();
        if (data.ok) {
          updateDetectorButtons(data.detector_mode);
          statusEl.textContent = 'Detektor: ' + data.detector_mode;
          statusEl.className = 'ok';
        } else {
          statusEl.textContent = 'Fehler: ' + (data.error || 'unknown');
          statusEl.className = 'warn';
        }
      } catch (e) {
        statusEl.textContent = 'Detektor-Wechsel fehlgeschlagen';
        statusEl.className = 'warn';
      } finally {
        document.querySelectorAll('.det-btn').forEach(b => b.disabled = false);
      }
    }
    document.getElementById('det_local').onclick = () => setDetectorMode('local');
    document.getElementById('det_api').onclick = () => setDetectorMode('api');
    document.getElementById('det_hybrid').onclick = () => setDetectorMode('hybrid');

    setInterval(pollState, 1000);
    setInterval(drawZone, 1000);
  </script>
</body>
</html>
"""


class DashStreamer:
  def __init__(
    self,
    output_dir: str,
    fps: float,
    segment_time: float = 1.0,
    list_size: int = 12,
    x264_preset: str = "veryfast",
    x264_tune: str = "zerolatency",
    x264_crf: int = 28,
    hwaccel: str = "auto",
    vaapi_device: str = "/dev/dri/renderD128",
    abr_enabled: bool = True,
    abr_high_bitrate_kbps: int = 1400,
    abr_low_bitrate_kbps: int = 650,
    abr_low_scale: float = 0.6,
  ):
    self.output_dir = output_dir
    self.fps = max(1.0, float(fps))
    self.segment_time = max(0.5, float(segment_time))
    self.list_size = max(3, int(list_size))
    self.x264_preset = str(x264_preset or "veryfast")
    self.x264_tune = str(x264_tune or "zerolatency")
    self.x264_crf = max(18, min(45, int(x264_crf)))
    self.hwaccel = str(hwaccel or "auto").lower()
    self.vaapi_device = str(vaapi_device or "/dev/dri/renderD128")
    self.abr_enabled = bool(abr_enabled)
    self.abr_high_bitrate_kbps = max(300, int(abr_high_bitrate_kbps))
    self.abr_low_bitrate_kbps = max(150, min(self.abr_high_bitrate_kbps - 50, int(abr_low_bitrate_kbps)))
    self.abr_low_scale = max(0.3, min(0.9, float(abr_low_scale)))
    self.encoder_name = "libx264"
    self._process: Optional[subprocess.Popen] = None
    self._ffmpeg_executable = self._resolve_ffmpeg_executable()
    self._enabled = self._ffmpeg_executable is not None
    self._lock = threading.Lock()
    self._preview_path = os.path.join(self.output_dir, "preview.jpg")

  @staticmethod
  def _resolve_ffmpeg_executable() -> Optional[str]:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
      return system_ffmpeg
    try:
      import imageio_ffmpeg

      return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
      return None

  @property
  def enabled(self) -> bool:
    return self._enabled

  @property
  def preview_path(self) -> str:
    return self._preview_path

  def start(self):
    if not self._enabled:
      return
    os.makedirs(self.output_dir, exist_ok=True)
    for name in os.listdir(self.output_dir):
      if name.endswith(".m4s") or name.endswith(".mpd") or name.endswith(".tmp"):
        try:
          os.remove(os.path.join(self.output_dir, name))
        except OSError:
          pass

    manifest_path = os.path.join(self.output_dir, "stream.mpd")
    gop = max(10, int(round(self.fps * 2.0)))

    base_input = [
      self._ffmpeg_executable,
      "-hide_banner",
      "-loglevel",
      "error",
      "-y",
      "-f",
      "mjpeg",
      "-r",
      str(self.fps),
      "-i",
      "pipe:0",
      "-an",
    ]
    dash_output = [
      "-g",
      str(gop),
      "-keyint_min",
      str(gop),
      "-sc_threshold",
      "0",
      "-use_timeline",
      "1",
      "-use_template",
      "1",
      "-window_size",
      str(self.list_size),
      "-extra_window_size",
      "1",
      "-seg_duration",
      str(self.segment_time),
      "-f",
      "dash",
      manifest_path,
    ]

    candidates = []
    use_vaapi = self.hwaccel in {"auto", "vaapi"} and os.path.exists(self.vaapi_device)
    if use_vaapi:
      candidates.append(
        (
          "h264_vaapi",
          [
            self._ffmpeg_executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-vaapi_device",
            self.vaapi_device,
            "-f",
            "mjpeg",
            "-r",
            str(self.fps),
            "-i",
            "pipe:0",
            "-an",
            "-vf",
            "format=nv12,hwupload",
            "-c:v",
            "h264_vaapi",
            "-qp",
            "26",
            *dash_output,
          ],
        )
      )

    if self.abr_enabled:
      low_scale_expr = f"trunc(iw*{self.abr_low_scale:.3f}/2)*2:trunc(ih*{self.abr_low_scale:.3f}/2)*2"
      candidates.append(
        (
          "libx264_abr",
          [
            *base_input,
            "-filter_complex",
            f"[0:v]split=2[v0][v1];[v1]scale={low_scale_expr}[v1s]",
            "-map",
            "[v0]",
            "-map",
            "[v1s]",
            "-c:v:0",
            "libx264",
            "-preset:v:0",
            self.x264_preset,
            "-tune:v:0",
            self.x264_tune,
            "-b:v:0",
            f"{self.abr_high_bitrate_kbps}k",
            "-maxrate:v:0",
            f"{int(self.abr_high_bitrate_kbps * 1.25)}k",
            "-bufsize:v:0",
            f"{int(self.abr_high_bitrate_kbps * 2)}k",
            "-g:v:0",
            str(gop),
            "-keyint_min:v:0",
            str(gop),
            "-sc_threshold:v:0",
            "0",
            "-pix_fmt:v:0",
            "yuv420p",
            "-c:v:1",
            "libx264",
            "-preset:v:1",
            self.x264_preset,
            "-tune:v:1",
            self.x264_tune,
            "-b:v:1",
            f"{self.abr_low_bitrate_kbps}k",
            "-maxrate:v:1",
            f"{int(self.abr_low_bitrate_kbps * 1.25)}k",
            "-bufsize:v:1",
            f"{int(self.abr_low_bitrate_kbps * 2)}k",
            "-g:v:1",
            str(gop),
            "-keyint_min:v:1",
            str(gop),
            "-sc_threshold:v:1",
            "0",
            "-pix_fmt:v:1",
            "yuv420p",
            "-adaptation_sets",
            "id=0,streams=v",
            *dash_output,
          ],
        )
      )

    candidates.append(
      (
        "libx264",
        [
          *base_input,
          "-c:v",
          "libx264",
          "-preset",
          self.x264_preset,
          "-tune",
          self.x264_tune,
          "-crf",
          str(self.x264_crf),
          "-pix_fmt",
          "yuv420p",
          *dash_output,
        ],
      )
    )

    self._process = None
    for encoder_name, command in candidates:
      process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
      )
      time.sleep(0.15)
      if process.poll() is None:
        self._process = process
        self.encoder_name = encoder_name
        break
      try:
        process.kill()
      except Exception:
        pass
      try:
        process.wait(timeout=0.2)
      except Exception:
        pass

    if self._process is None:
      self._enabled = False

  def write_jpeg(self, jpeg: bytes):
    if not self._enabled or self._process is None or self._process.stdin is None:
      return
    with self._lock:
      if self._process.poll() is not None:
        return
      try:
        self._process.stdin.write(jpeg)
        self._process.stdin.flush()
      except Exception:
        pass

  def write_preview(self, jpeg: bytes):
    try:
      os.makedirs(self.output_dir, exist_ok=True)
      with open(self._preview_path, "wb") as handle:
        handle.write(jpeg)
    except Exception:
      pass

  def stop(self):
    if self._process is None:
      return
    try:
      if self._process.stdin:
        self._process.stdin.close()
    except Exception:
      pass
    try:
      self._process.terminate()
      self._process.wait(timeout=1)
    except Exception:
      try:
        self._process.kill()
      except Exception:
        pass
    self._process = None


class TrackingEngine:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = os.path.abspath(config_path)
        self.config_dir = os.path.dirname(self.config_path)
        self.config_manager = ConfigManager(config_path=self.config_path)
        self.config = self.config_manager.load()

        tracking_cfg = self.config["tracking"]
        tracker_path = self._resolve_config_relative_path(str(tracking_cfg["tracker"]))

        self.detector = build_person_detector(
          tracking_cfg=tracking_cfg,
          config_dir=self.config_dir,
        )
        self.video_mode = self._sanitize_video_mode(self.config.get("video", {}).get("input_mode", "youtube"))
        self.simulation_control_mode = self._sanitize_simulation_control_mode(
          self.config.get("video", {}).get("simulation", {}).get("control_mode", "remote_control")
        )
        self.simulation_idle_loop = self._sanitize_simulation_idle_loop(
          self.config.get("video", {}).get("simulation", {}).get("idle_loop", False)
        )
        simulation_dir = str(
          self.config.get("video", {}).get("simulation", {}).get("directory", DEFAULT_SIMULATION_DIR)
        ).strip() or DEFAULT_SIMULATION_DIR
        self.simulation_registry = SimulationVideoRegistry(self._resolve_config_relative_path(simulation_dir))
        self.simulation_registry.refresh()
        self._simulation_pending_clip_ids = deque()
        self._simulation_basic_clip_id = ""
        self._simulation_active_clip_id = ""
        self._simulation_active_kind = "basic"
        self._simulation_lock = threading.Lock()
        self._initialize_simulation_state(self.config.get("video", {}))
        self.video_input = self._create_video_input(self.config.get("video", {}))

        self.tracking_module = YOLOTrackingModule(
            detector=self.detector,
            tracker_config=tracker_path,
            confidence_threshold=float(tracking_cfg["confidence_threshold"]),
            iou_threshold=float(tracking_cfg["iou_threshold"]),
            image_size=int(tracking_cfg["imgsz"]),
            tta_enabled=bool(tracking_cfg.get("tta_enabled", False)),
            max_detections=int(tracking_cfg.get("max_detections", 300)),
            stabilization_enabled=bool(tracking_cfg.get("stabilization_enabled", True)),
            track_hold_frames=int(tracking_cfg.get("track_hold_frames", 5)),
            box_ema_alpha=float(tracking_cfg.get("box_ema_alpha", 0.65)),
            hold_confidence_decay=float(tracking_cfg.get("hold_confidence_decay", 0.85)),
            trail_length=int(tracking_cfg.get("trail_length", 12)),
            motion_min_pixels=float(tracking_cfg.get("motion_min_pixels", 2.0)),
          preprocess_enabled=bool(self.config.get("preprocess", {}).get("enabled", False)),
          preprocess_upscale=float(self.config.get("preprocess", {}).get("upscale", 1.0)),
          preprocess_clahe_clip=float(self.config.get("preprocess", {}).get("clahe_clip", 2.0)),
          preprocess_denoise=bool(self.config.get("preprocess", {}).get("denoise", False)),
        )
        self.process_every_n_frames = max(1, int(tracking_cfg.get("process_every_n_frames", 1)))
        self.analysis_roi = self._normalize_analysis_roi(tracking_cfg.get("analysis_roi", {}))

        self.zone_config = EntranceZoneConfig.from_dict(self.config["zone"])
        self.entry_analysis = TrajectoryEntryAnalysisModule(
          zone_config=self.zone_config,
          reid_max_gap_frames=int(tracking_cfg.get("reid_max_gap_frames", 24)),
          reid_max_distance_px=float(tracking_cfg.get("reid_max_distance_px", 150.0)),
          reid_ambiguity_ratio=float(tracking_cfg.get("reid_ambiguity_ratio", 0.87)),
        )
        self.occupancy_state = OccupancyStateModule(db=None)
        self.visualizer = VisualizationOutputModule(show_window=False, enable_zone_editor=False)
        self.integration_writer: Optional[PrognoseDbWriter] = None
        self.integration_write_mode = "occupancy_state"

        integration_cfg = dict(self.config.get("integration", {}) or {})
        prognose_cfg = integration_cfg.get("prognose_db", integration_cfg)
        if isinstance(prognose_cfg, dict) and prognose_cfg.get("enabled", False):
          try:
            self.integration_writer = PrognoseDbWriter(
              config=prognose_cfg,
              config_dir=self.config_dir,
              component_name="website-dashboard.realtime.TrackingEngine",
            )
            self.integration_write_mode = str(
              prognose_cfg.get("write_mode", getattr(self.integration_writer, "write_mode", "frame_near"))
            ).strip().lower() or "frame_near"
          except Exception:
            self.integration_writer = None

        dashboard_cfg = self.config.get("dashboard", {})
        self.stream_fps = float(dashboard_cfg.get("stream_fps", 25.0))
        self.capture_max_fps = max(0.0, float(dashboard_cfg.get("capture_max_fps", 0.0)))
        self.capture_frame_interval = (1.0 / self.capture_max_fps) if self.capture_max_fps > 0 else 0.0
        self.visual_update_fps = max(0.0, float(dashboard_cfg.get("visual_update_fps", self.stream_fps)))
        self.visual_update_interval = (1.0 / self.visual_update_fps) if self.visual_update_fps > 0 else 0.0
        self.jpeg_quality = max(20, min(95, int(dashboard_cfg.get("jpeg_quality", 80))))
        self.jpeg_optimize = bool(dashboard_cfg.get("jpeg_optimize", False))
        self.stream_max_width = max(320, int(dashboard_cfg.get("stream_max_width", 960)))
        dash_cfg = dashboard_cfg.get("dash", dashboard_cfg.get("hls", {}))
        self.dash_enabled = bool(dash_cfg.get("enabled", True))
        self.dash_output_dir = self._resolve_config_relative_path(
          str(dash_cfg.get("output_dir", "runtime/dash"))
        )
        self.dash_segment_time = float(dash_cfg.get("segment_time", 1.0))
        self.dash_list_size = int(dash_cfg.get("list_size", 12))
        self.dash_x264_preset = str(dash_cfg.get("preset", "veryfast"))
        self.dash_x264_tune = str(dash_cfg.get("tune", "zerolatency"))
        self.dash_x264_crf = int(dash_cfg.get("crf", 28))
        self.dash_hwaccel = str(dash_cfg.get("hwaccel", "auto"))
        self.dash_vaapi_device = str(dash_cfg.get("vaapi_device", "/dev/dri/renderD128"))
        dash_abr_cfg = dash_cfg.get("abr", {})
        self.dash_abr_enabled = bool(dash_abr_cfg.get("enabled", True))
        self.dash_abr_high_bitrate_kbps = int(dash_abr_cfg.get("high_bitrate_kbps", 1400))
        self.dash_abr_low_bitrate_kbps = int(dash_abr_cfg.get("low_bitrate_kbps", 650))
        self.dash_abr_low_scale = float(dash_abr_cfg.get("low_scale", 0.6))
        self.analysis_queue_frames = max(8, int(dashboard_cfg.get("analysis_queue_frames", 64)))
        self.analysis_skip_threshold_frames = max(0, int(dashboard_cfg.get("analysis_skip_threshold_frames", 0)))
        self.visual_stale_fallback_sec = max(0.0, float(dashboard_cfg.get("visual_stale_fallback_sec", 0.5)))
        self.dynamic_skip_enabled = bool(dashboard_cfg.get("dynamic_skip_enabled", True))
        self.dynamic_skip_queue_threshold = max(1, int(dashboard_cfg.get("dynamic_skip_queue_threshold", 6)))
        self.dynamic_skip_max_n = max(
          self.process_every_n_frames,
          int(dashboard_cfg.get("dynamic_skip_max_n", max(4, self.process_every_n_frames))),
        )
        self.dash_streamer = DashStreamer(
          output_dir=self.dash_output_dir,
          fps=self.stream_fps,
          segment_time=self.dash_segment_time,
          list_size=self.dash_list_size,
          x264_preset=self.dash_x264_preset,
          x264_tune=self.dash_x264_tune,
          x264_crf=self.dash_x264_crf,
          hwaccel=self.dash_hwaccel,
          vaapi_device=self.dash_vaapi_device,
          abr_enabled=self.dash_abr_enabled,
          abr_high_bitrate_kbps=self.dash_abr_high_bitrate_kbps,
          abr_low_bitrate_kbps=self.dash_abr_low_bitrate_kbps,
          abr_low_scale=self.dash_abr_low_scale,
        )
        if not self.dash_enabled:
          self.dash_streamer._enabled = False

        self.entries_total = 0
        self.exits_total = 0
        self.last_tracks = 0
        self.current_frame_occupancy = 0
        self.last_entries = 0
        self.last_exits = 0
        self.last_fps = 0.0
        self.last_inference_fps = 0.0
        self.last_track_ok = True
        self.last_track_error = ""
        self._fps_ema = 0.0
        self.analysis_skipped_frames = 0
        self._pause_write_heartbeat_sec = 1.0
        self._last_pause_write_ts = 0.0

        self._lock = threading.Lock()
        self._packet_cv = threading.Condition(self._lock)
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._inference_thread: Optional[threading.Thread] = None
        self._packetizer_thread: Optional[threading.Thread] = None
        self._capture_frame_idx = 0
        self._last_tracks_cache = []
        self._analysis_queue = deque()
        self._latest_visual_jpeg: Optional[bytes] = None
        self._latest_raw_jpeg: Optional[bytes] = None
        self._latest_visual_ts = 0.0
        self._latest_raw_ts = 0.0
        self._dash_frame_counter = 0
        self._next_capture_deadline_ts = 0.0
        self._last_visual_update_ts = 0.0
        self._current_effective_process_n = self.process_every_n_frames
        self._track_event_overlay: Dict[int, Dict[str, Any]] = {}
        self._track_event_overlay_ttl_sec = 2.0
        self._video_input_lock = threading.Lock()
        self.analysis_enabled = True

        profile_cfg = dict(dashboard_cfg.get("profile_simulation", {}) or {})
        excel_path = str(profile_cfg.get("excel_path", "KI_Projekt_Daten_einJahr.xlsx")).strip() or "KI_Projekt_Daten_einJahr.xlsx"
        if not os.path.isabs(excel_path):
          excel_path = self._resolve_config_relative_path(excel_path)
        self.profile_simulation = ProfileOccupancySimulation(
          excel_path=excel_path,
          enabled=bool(profile_cfg.get("enabled", False)),
          tick_seconds=float(profile_cfg.get("tick_seconds", 60.0)),
          profile_blend=float(profile_cfg.get("profile_blend", 0.72)),
          noise_sigma_scale=float(profile_cfg.get("noise_sigma_scale", 0.85)),
          max_step_per_tick=float(profile_cfg.get("max_step_per_tick", 2.0)),
          rollback_minutes=float(profile_cfg.get("rollback_minutes", 15.0)),
        )

    def start(self):
      if self._running:
        return
      with self._video_input_lock:
        opened = self.video_input.open()
      if not opened:
        print("[WARN] Videoquelle beim Start nicht geöffnet; reconnect läuft im Hintergrund weiter.")
      self.dash_streamer.start()
      self._running = True
      self.profile_simulation.start()
      self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
      self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
      self._packetizer_thread = threading.Thread(target=self._packetizer_loop, daemon=True)
      self._capture_thread.start()
      self._inference_thread.start()
      self._packetizer_thread.start()

    def stop(self):
      self._running = False
      self.profile_simulation.stop()
      if self._capture_thread:
        self._capture_thread.join(timeout=2)
      if self._inference_thread:
        self._inference_thread.join(timeout=2)
      if self._packetizer_thread:
        self._packetizer_thread.join(timeout=2)
      self.dash_streamer.stop()
      with self._video_input_lock:
        self.video_input.release()
      if self.integration_writer:
        self.integration_writer.close()

    @staticmethod
    def _sanitize_video_mode(value: Any) -> str:
      mode = str(value or "youtube").strip().lower()
      if mode not in {"youtube", "livefeed_simulation"}:
        return "youtube"
      return mode

    @staticmethod
    def _sanitize_simulation_control_mode(value: Any) -> str:
      mode = str(value or "remote_control").strip().lower()
      if mode not in {"remote_control", "auto_rules"}:
        return "remote_control"
      return mode

    @staticmethod
    def _sanitize_simulation_idle_loop(value: Any) -> bool:
      if isinstance(value, bool):
        return value
      if isinstance(value, (int, float)):
        return bool(value)
      normalized = str(value or "").strip().lower()
      if normalized in {"1", "true", "yes", "on"}:
        return True
      if normalized in {"0", "false", "no", "off"}:
        return False
      return False

    @staticmethod
    def _count_active_tracks(tracks: list[dict[str, Any]] | None) -> int:
      if not isinstance(tracks, list):
        return 0
      active_count = 0
      for track in tracks:
        if not isinstance(track, dict):
          continue
        if bool(track.get("is_stale", False)):
          continue
        active_count += 1
      return active_count

    def _resolve_live_occupancy(self, tracks: list[dict[str, Any]] | None = None) -> int:
      if self.profile_simulation.is_enabled():
        return int(self.profile_simulation.current_occupancy())
      if self.integration_write_mode == "frame_near":
        if tracks is not None:
          return self._count_active_tracks(tracks)
        return int(self.current_frame_occupancy)
      return int(self.occupancy_state.occupancy)

    def _accept_temporal_detection_event(self, event: Dict[str, Any]) -> bool:
      event_type = str(event.get("type", "entry")).lower()
      if event_type not in {"entry", "exit"}:
        return False

      track_id = event.get("track_id")
      if track_id is None:
        return False

      try:
        normalized_track_id = int(track_id)
      except (TypeError, ValueError):
        return False

      last_type = self.occupancy_state.last_event_type_by_track.get(normalized_track_id)
      if last_type == event_type:
        return False

      self.occupancy_state.last_event_type_by_track[normalized_track_id] = event_type
      self.occupancy_state.last_event_time = datetime.now()
      return True

    def _ensure_simulation_defaults(self, video_cfg: Dict[str, Any]) -> Dict[str, Any]:
      merged = dict(video_cfg or {})
      simulation_cfg = dict(merged.get("simulation", {}) or {})
      simulation_cfg["control_mode"] = self._sanitize_simulation_control_mode(
        simulation_cfg.get("control_mode", self.simulation_control_mode)
      )
      simulation_cfg["idle_loop"] = self._sanitize_simulation_idle_loop(
        simulation_cfg.get("idle_loop", self.simulation_idle_loop)
      )
      simulation_cfg["directory"] = str(simulation_cfg.get("directory", DEFAULT_SIMULATION_DIR)).strip() or DEFAULT_SIMULATION_DIR
      simulation_cfg["default_clip_id"] = str(simulation_cfg.get("default_clip_id", "")).strip()
      merged["simulation"] = simulation_cfg
      return merged

    def _initialize_simulation_state(self, video_cfg: Dict[str, Any]):
      prepared_cfg = self._ensure_simulation_defaults(video_cfg)
      self.simulation_registry.refresh()
      basic_clip = self._resolve_basic_simulation_clip(prepared_cfg)
      with self._simulation_lock:
        self._simulation_pending_clip_ids.clear()
        if basic_clip:
          self._simulation_basic_clip_id = basic_clip.clip_id
          self._simulation_active_clip_id = basic_clip.clip_id
          self._simulation_active_kind = "basic"

    def _resolve_basic_simulation_clip(self, video_cfg: Dict[str, Any]):
      simulation_cfg = dict(video_cfg.get("simulation", {}) or {})
      default_clip_id = str(simulation_cfg.get("default_clip_id", "")).strip()
      if default_clip_id:
        clip = self.simulation_registry.get_clip(default_clip_id)
        if clip:
          return clip

      basic_loop = self.simulation_registry.get_default_basic_loop()
      if basic_loop:
        return basic_loop

      clips = self.simulation_registry.list_clips()
      if clips:
        first_id = str(clips[0].get("clip_id", "")).strip()
        if first_id:
          return self.simulation_registry.get_clip(first_id)
      return None

    def _resolve_simulation_source(self, video_cfg: Dict[str, Any]) -> str:
      simulation_cfg = dict(video_cfg.get("simulation", {}) or {})
      self.simulation_control_mode = self._sanitize_simulation_control_mode(simulation_cfg.get("control_mode", "remote_control"))
      self.simulation_idle_loop = self._sanitize_simulation_idle_loop(simulation_cfg.get("idle_loop", self.simulation_idle_loop))
      self.simulation_registry.refresh()

      with self._simulation_lock:
        active_clip_id = str(self._simulation_active_clip_id or "").strip()
      if active_clip_id:
        active_clip = self.simulation_registry.get_clip(active_clip_id)
        if active_clip:
          return active_clip.canonical_path

      basic_clip = self._resolve_basic_simulation_clip(video_cfg)
      if basic_clip:
        with self._simulation_lock:
          self._simulation_basic_clip_id = basic_clip.clip_id
          self._simulation_active_clip_id = basic_clip.clip_id
          self._simulation_active_kind = "basic"
        return basic_clip.canonical_path

      return "0"

    def _set_video_input_source(self, source_path: str, loop_file_source: bool = False) -> bool:
      replacement_input = VideoInputModule(
        source=source_path,
        fallback_source=None,
        reconnect_delay=float(self.config.get("video", {}).get("reconnect_delay", 1.0)),
        max_retries=int(self.config.get("video", {}).get("max_retries", 0)),
        loop_file_source=loop_file_source,
        hwaccel=str(self.config.get("video", {}).get("hwaccel", "auto")),
        youtube_cookies_from_browser=str(self.config.get("video", {}).get("youtube_cookies_from_browser", "")),
        youtube_cookiefile=str(self.config.get("video", {}).get("youtube_cookiefile", "")),
        youtube_format=str(self.config.get("video", {}).get("youtube_format", "best[ext=mp4]/best")),
        youtube_player_client=str(self.config.get("video", {}).get("youtube_player_client", "android")),
      )

      with self._video_input_lock:
        previous_input = self.video_input
        self.video_input = replacement_input
        try:
          previous_input.release()
        except Exception:
          pass
        return bool(self.video_input.open())

    def _switch_to_simulation_clip_id(self, clip_id: str, kind: str) -> bool:
      clip = self.simulation_registry.get_clip(clip_id)
      if not clip:
        self.simulation_registry.refresh()
        clip = self.simulation_registry.get_clip(clip_id)
      if not clip:
        return False

      loop_file_source = bool(kind == "basic" and self.simulation_idle_loop)
      opened = self._set_video_input_source(clip.canonical_path, loop_file_source=loop_file_source)
      if opened:
        with self._simulation_lock:
          self._simulation_active_clip_id = clip.clip_id
          self._simulation_active_kind = kind
      return opened

    def _simulation_switch_after_eof(self) -> bool:
      with self._simulation_lock:
        next_clip_id = self._simulation_pending_clip_ids.popleft() if self._simulation_pending_clip_ids else ""
        active_kind = self._simulation_active_kind
        basic_clip_id = self._simulation_basic_clip_id
        idle_loop = bool(self.simulation_idle_loop)

      if next_clip_id:
        switched = self._switch_to_simulation_clip_id(next_clip_id, kind="scheduled")
        if switched:
          return True

      if active_kind == "scheduled" and basic_clip_id:
        switched = self._switch_to_simulation_clip_id(basic_clip_id, kind="basic")
        if switched:
          return True

      if active_kind == "basic":
        if idle_loop and basic_clip_id:
          return bool(self._switch_to_simulation_clip_id(basic_clip_id, kind="basic"))
        return False

      if basic_clip_id:
        return bool(self._switch_to_simulation_clip_id(basic_clip_id, kind="basic"))
      return False

    def _simulation_clip_brief(self, clip_id: str) -> Dict[str, Any]:
      normalized_id = str(clip_id or "").strip()
      if not normalized_id:
        return {"clip_id": "", "display_name": "-", "relative_path": ""}

      clip = self.simulation_registry.get_clip(normalized_id)
      if not clip:
        self.simulation_registry.refresh()
        clip = self.simulation_registry.get_clip(normalized_id)

      if not clip:
        return {
          "clip_id": normalized_id,
          "display_name": normalized_id,
          "relative_path": "",
        }

      return {
        "clip_id": clip.clip_id,
        "display_name": clip.display_name,
        "relative_path": clip.relative_path,
      }

    def _simulation_queue_snapshot_locked(self) -> List[Dict[str, Any]]:
      return [
        self._simulation_clip_brief(clip_id)
        for clip_id in list(self._simulation_pending_clip_ids)
      ]

    def _create_video_input(self, video_cfg: Dict[str, Any]) -> VideoInputModule:
      prepared_cfg = self._ensure_simulation_defaults(video_cfg)
      source = str(prepared_cfg.get("source", "0"))
      loop_file_source = True
      if self.video_mode == "livefeed_simulation":
        source = self._resolve_simulation_source(prepared_cfg)
        loop_file_source = False
      cookiefile = str(video_cfg.get("youtube_cookiefile", "") or "").strip()
      if cookiefile and not os.path.isabs(cookiefile):
        cookiefile = os.path.abspath(os.path.join(self.config_dir, cookiefile))
      return VideoInputModule(
        source=source,
        fallback_source=None,
        reconnect_delay=float(video_cfg.get("reconnect_delay", 1.0)),
        max_retries=int(video_cfg.get("max_retries", 0)),
        loop_file_source=loop_file_source,
        hwaccel=str(video_cfg.get("hwaccel", "auto")),
        youtube_cookies_from_browser=str(video_cfg.get("youtube_cookies_from_browser", "")),
        youtube_cookiefile=cookiefile,
        youtube_format=str(video_cfg.get("youtube_format", "best[ext=mp4]/best")),
        youtube_player_client=str(video_cfg.get("youtube_player_client", "android")),
      )

    def _resolve_config_relative_path(self, path_value: str) -> str:
      if os.path.isabs(path_value):
        return path_value
      return os.path.abspath(os.path.join(self.config_dir, path_value))

    def _encode_stream_jpeg(self, frame):
      encode_frame = frame
      frame_h, frame_w = frame.shape[:2]
      if frame_w > self.stream_max_width:
        scale = self.stream_max_width / float(frame_w)
        target_h = max(1, int(frame_h * scale))
        encode_frame = cv2.resize(frame, (self.stream_max_width, target_h), interpolation=cv2.INTER_AREA)

      return cv2.imencode(
        ".jpg",
        encode_frame,
        [
          int(cv2.IMWRITE_JPEG_QUALITY),
          self.jpeg_quality,
          int(cv2.IMWRITE_JPEG_OPTIMIZE),
          1 if self.jpeg_optimize else 0,
        ],
      )

    def _capture_loop(self):
      while self._running:
        if self.capture_frame_interval > 0:
          now_ts = time.monotonic()
          if self._next_capture_deadline_ts <= 0:
            self._next_capture_deadline_ts = now_ts
          sleep_for = self._next_capture_deadline_ts - now_ts
          if sleep_for > 0:
            time.sleep(min(sleep_for, 0.02))

        frame = None
        with self._video_input_lock:
          ok, frame = self.video_input.read()
        if not ok or frame is None:
          if self.video_mode == "livefeed_simulation":
            switched = self._simulation_switch_after_eof()
            if not switched:
              time.sleep(0.12)
              continue
          time.sleep(0.02)
          continue

        if self.capture_frame_interval > 0:
          now_ts = time.monotonic()
          if self._next_capture_deadline_ts <= 0:
            self._next_capture_deadline_ts = now_ts + self.capture_frame_interval
          else:
            while self._next_capture_deadline_ts <= now_ts:
              self._next_capture_deadline_ts += self.capture_frame_interval

        with self._lock:
          if self.analysis_enabled:
            if len(self._analysis_queue) >= self.analysis_queue_frames:
              try:
                self._analysis_queue.popleft()
                self.analysis_skipped_frames += 1
              except IndexError:
                pass

            self._capture_frame_idx += 1
            frame_id = self._capture_frame_idx
            self._analysis_queue.append((frame_id, frame))
          else:
            self._analysis_queue.clear()

        should_update_raw = False
        now_ts = time.monotonic()
        with self._packet_cv:
          if self._latest_visual_jpeg is None:
            should_update_raw = True
          else:
            visual_age = now_ts - self._latest_visual_ts
            if visual_age > self.visual_stale_fallback_sec:
              should_update_raw = True

        if should_update_raw:
          ok_jpeg, encoded = self._encode_stream_jpeg(frame)
          if ok_jpeg:
            with self._packet_cv:
              self._latest_raw_jpeg = encoded.tobytes()
              self._latest_raw_ts = time.monotonic()
              self._packet_cv.notify_all()

    def _inference_loop(self):
      while self._running:
        t0 = time.time()

        with self._lock:
          analysis_enabled = bool(self.analysis_enabled)

        if not analysis_enabled:
          sim_tick_events = self.profile_simulation.consume_tick_events()
          sim_entries = int(sim_tick_events.get("entries", 0))
          sim_exits = int(sim_tick_events.get("exits", 0))
          if sim_entries > 0:
            self.entries_total += sim_entries
          if sim_exits > 0:
            self.exits_total += sim_exits

          live_occupancy = self._resolve_live_occupancy()
          frame_id = self._capture_frame_idx
          now_pause = time.monotonic()
          should_write_pause_frame = (sim_entries > 0 or sim_exits > 0)
          if not should_write_pause_frame:
            should_write_pause_frame = (now_pause - self._last_pause_write_ts) >= self._pause_write_heartbeat_sec

          if self.integration_writer and should_write_pause_frame:
            # Keep downstream forecast ingestion active in pause-mode demos.
            self.integration_writer.write_frame(
              occupancy=live_occupancy,
              tracks=[],
              run_tracking_now=False,
              track_ok=True,
              track_error="",
              model_name=str(getattr(self.detector, "model_name", "yolo")),
              model_version=str(getattr(self.detector, "model_version", "unknown")),
              frame_id=frame_id,
              events_in_frame={"entry": sim_entries, "exit": sim_exits},
            )
            self._last_pause_write_ts = now_pause

          self.last_tracks = 0
          self.last_entries = sim_entries
          self.last_exits = sim_exits
          self.last_inference_fps = 0.0
          self.last_track_ok = True
          self.last_track_error = ""
          self._last_tracks_cache = []
          time.sleep(0.02)
          continue

        with self._lock:
          if not self._analysis_queue:
            frame_tuple = None
            queue_len = 0
          else:
            if self.analysis_skip_threshold_frames > 0:
              while len(self._analysis_queue) > self.analysis_skip_threshold_frames:
                self._analysis_queue.popleft()
                self.analysis_skipped_frames += 1
            queue_len = len(self._analysis_queue)
            frame_tuple = self._analysis_queue.popleft() if self._analysis_queue else None

        if frame_tuple is None:
          time.sleep(0.005)
          continue

        frame_id, frame = frame_tuple
        effective_process_n = self.process_every_n_frames
        if self.dynamic_skip_enabled and queue_len > self.dynamic_skip_queue_threshold:
          pressure_ratio = queue_len / float(self.dynamic_skip_queue_threshold)
          extra_skip = int(pressure_ratio)
          effective_process_n = min(
            self.dynamic_skip_max_n,
            self.process_every_n_frames + max(1, extra_skip),
          )

        self._current_effective_process_n = effective_process_n
        run_tracking_now = (frame_id % effective_process_n) == 0

        if run_tracking_now:
          infer_t0 = time.time()
          analysis_frame, roi_offset = self._crop_to_analysis_roi(frame)
          tracks = self.tracking_module.track(analysis_frame)
          self.last_track_ok = bool(getattr(self.detector, "last_track_ok", True))
          self.last_track_error = str(getattr(self.detector, "last_track_error", "") or "")
          if roi_offset != (0, 0):
            tracks = self._remap_tracks_to_full_frame(tracks, roi_offset)
          self._last_tracks_cache = tracks
          infer_elapsed = max(1e-6, time.time() - infer_t0)
          if self.last_track_ok:
            raw_infer_fps = 1.0 / infer_elapsed
            cap_fps = max(1.0, self.capture_max_fps if self.capture_max_fps > 0 else self.stream_fps)
            self.last_inference_fps = round(min(raw_infer_fps, cap_fps), 2)
          else:
            self.last_inference_fps = 0.0
        else:
          tracks = self._last_tracks_cache

        frame_occupancy = self._count_active_tracks(tracks)
        self.current_frame_occupancy = frame_occupancy
        live_occupancy = self._resolve_live_occupancy(tracks)

        sim_tick_events = self.profile_simulation.consume_tick_events()
        sim_entries = int(sim_tick_events.get("entries", 0))
        sim_exits = int(sim_tick_events.get("exits", 0))
        if sim_entries > 0:
          self.entries_total += sim_entries
        if sim_exits > 0:
          self.exits_total += sim_exits

        events = self.entry_analysis.update(tracks=tracks, frame_shape=frame.shape) if run_tracking_now else []
        frame_entries = sim_entries
        frame_exits = sim_exits
        now_overlay = time.monotonic()
        profile_enabled = self.profile_simulation.is_enabled()
        for event in events:
          accepted = self._accept_temporal_detection_event(event) if profile_enabled else self.occupancy_state.handle_event(event)
          if accepted:
            event_track_id = event.get("track_id")
            if event_track_id is not None:
              try:
                self._track_event_overlay[int(event_track_id)] = {
                  "type": str(event.get("type", "entry")).lower(),
                  "ts": now_overlay,
                }
              except (TypeError, ValueError):
                pass
            if str(event.get("type", "entry")).lower() == "entry":
              if profile_enabled:
                self.profile_simulation.register_detection_event("entry")
              self.entries_total += 1
              frame_entries += 1
            elif str(event.get("type", "entry")).lower() == "exit":
              if profile_enabled:
                self.profile_simulation.register_detection_event("exit")
              self.exits_total += 1
              frame_exits += 1

        if self._track_event_overlay:
          expired_ids = [
            track_id
            for track_id, payload in self._track_event_overlay.items()
            if (now_overlay - float(payload.get("ts", 0.0))) > self._track_event_overlay_ttl_sec
          ]
          for track_id in expired_ids:
            self._track_event_overlay.pop(track_id, None)

        if self.integration_writer:
          self.integration_writer.write_frame(
            occupancy=live_occupancy,
            tracks=tracks,
            run_tracking_now=run_tracking_now,
            track_ok=self.last_track_ok,
            track_error=self.last_track_error,
            model_name=str(getattr(self.detector, "model_name", "yolo")),
            model_version=str(getattr(self.detector, "model_version", "unknown")),
            frame_id=frame_id,
            events_in_frame={"entry": frame_entries, "exit": frame_exits},
          )

        now_visual = time.monotonic()
        should_update_visual = True

        if should_update_visual:
          output = self.visualizer.draw(
            frame=frame,
            tracks=tracks,
            zone_config=self.zone_config,
            occupancy=live_occupancy,
            entries_total=self.entries_total,
            exits_total=self.exits_total,
            events_in_frame={"entry": frame_entries, "exit": frame_exits},
            track_event_overlay=self._track_event_overlay,
            analysis_roi=self.analysis_roi,
          )

          ok_jpeg, encoded = self._encode_stream_jpeg(output)
          if ok_jpeg:
            with self._packet_cv:
              self._latest_visual_jpeg = encoded.tobytes()
              self._latest_visual_ts = time.monotonic()
              self._packet_cv.notify_all()
            self._last_visual_update_ts = now_visual

        self.last_tracks = len(tracks)
        self.last_entries = frame_entries
        self.last_exits = frame_exits
        elapsed = max(1e-4, time.time() - t0)
        cap_fps = max(1.0, self.capture_max_fps if self.capture_max_fps > 0 else self.stream_fps)
        current_fps = min(cap_fps, 1.0 / elapsed)
        if self._fps_ema <= 0:
          self._fps_ema = current_fps
        else:
          self._fps_ema = (0.85 * self._fps_ema) + (0.15 * current_fps)
        self.last_fps = round(self._fps_ema, 2)

    def _packetizer_loop(self):
      packet_interval = 1.0 / max(1.0, self.stream_fps)
      next_deadline = time.monotonic()

      while self._running:
        sleep_for = next_deadline - time.monotonic()
        if sleep_for > 0:
          time.sleep(sleep_for)

        encoded_frame = None
        with self._packet_cv:
          now = time.monotonic()
          visual_age = now - self._latest_visual_ts if self._latest_visual_jpeg is not None else 10_000.0
          raw_age = now - self._latest_raw_ts if self._latest_raw_jpeg is not None else 10_000.0

          visual_valid = self._latest_visual_jpeg is not None and visual_age <= max(0.1, self.visual_stale_fallback_sec * 3.0)
          raw_valid = self._latest_raw_jpeg is not None and raw_age <= max(0.1, self.visual_stale_fallback_sec * 3.0)

          if visual_valid:
            encoded_frame = self._latest_visual_jpeg
          elif raw_valid:
            encoded_frame = self._latest_raw_jpeg
          elif self._latest_raw_jpeg is not None:
            encoded_frame = self._latest_raw_jpeg
          elif self._latest_visual_jpeg is not None:
            # Last-resort fallback when no fresh raw frame exists.
            encoded_frame = self._latest_visual_jpeg

        if encoded_frame is not None:
          self._dash_frame_counter += 1
          self.dash_streamer.write_jpeg(encoded_frame)
          if self._dash_frame_counter % max(1, int(self.stream_fps)) == 0:
            self.dash_streamer.write_preview(encoded_frame)

        next_deadline += packet_interval
        now = time.monotonic()
        if next_deadline < (now - packet_interval):
          next_deadline = now

    def get_state(self) -> Dict[str, Any]:
      with self._lock:
        # Drain pending simulation tick deltas so dashboard counters keep moving
        # even when analysis/inference is temporarily paused.
        sim_tick_events = self.profile_simulation.consume_tick_events()
        sim_entries = int(sim_tick_events.get("entries", 0))
        sim_exits = int(sim_tick_events.get("exits", 0))
        if sim_entries > 0:
          self.entries_total += sim_entries
        if sim_exits > 0:
          self.exits_total += sim_exits

        video_occupancy = int(self.current_frame_occupancy)
        simulated_occupancy = int(self.profile_simulation.current_occupancy())
        live_occupancy = self._resolve_live_occupancy()
        self.profile_simulation.observe_displayed_occupancy(live_occupancy)
        state = {
          "occupancy": live_occupancy,
          "displayed_occupancy": live_occupancy,
          "video_occupancy": video_occupancy,
          "simulated_occupancy": simulated_occupancy,
          "frame_occupancy": int(self.current_frame_occupancy),
          "session_occupancy": int(self.occupancy_state.occupancy),
          "occupancy_mode": self.integration_write_mode,
          "entries_total": self.entries_total,
          "exits_total": self.exits_total,
          "entries_frame": self.last_entries,
          "exits_frame": self.last_exits,
          "tracks": self.last_tracks,
          "fps": self.last_fps,
          "inference_fps": self.last_inference_fps,
          "analysis_queue": len(self._analysis_queue),
          "analysis_skipped_frames": self.analysis_skipped_frames,
          "capture_max_fps": self.capture_max_fps,
          "effective_process_every_n_frames": self._current_effective_process_n,
          "dash_enabled": self.dash_streamer.enabled,
          "dash_encoder": self.dash_streamer.encoder_name,
          "dash_output_dir": self.dash_output_dir,
          "track_ok": self.last_track_ok,
          "track_error": self.last_track_error,
          "detector_source": str(getattr(self.detector, "last_detector_source", "api")),
          "detector_mode": str(self.config.get("tracking", {}).get("detector_mode", "hybrid")),
          "zone": self.zone_config.to_dict(),
          "analysis_roi": self.analysis_roi,
          "integration_writer": (
            self.integration_writer.get_status() if self.integration_writer else {"enabled": False}
          ),
          "running": bool(self._running),
          "analysis_enabled": bool(self.analysis_enabled),
          "video_mode": self.video_mode,
          "simulation_control_mode": self.simulation_control_mode,
          "simulation_idle_loop": bool(self.simulation_idle_loop),
          "profile_simulation": self.profile_simulation.get_status(),
        }
      with self._video_input_lock:
        state["video_source"] = str(self.video_input.raw_source)
        state["video_active_source"] = str(getattr(self.video_input, "active_source", self.video_input.source))
        state["video_opened"] = bool(self.video_input.capture and self.video_input.capture.isOpened())
      with self._simulation_lock:
        state["simulation_active_clip_id"] = self._simulation_active_clip_id
        state["simulation_active_kind"] = self._simulation_active_kind
        state["simulation_basic_clip_id"] = self._simulation_basic_clip_id
        state["simulation_pending_count"] = len(self._simulation_pending_clip_ids)
        state["simulation_active_clip"] = self._simulation_clip_brief(self._simulation_active_clip_id)
        state["simulation_basic_clip"] = self._simulation_clip_brief(self._simulation_basic_clip_id)
        state["simulation_pending_queue"] = self._simulation_queue_snapshot_locked()
      return state

    def set_detector_mode(self, payload: Dict[str, Any]) -> Dict[str, Any]:
      mode = str(payload.get("mode", "")).strip().lower()
      if mode not in {"local", "api", "hybrid"}:
        raise ValueError(f"Ungültiger Modus: {mode}")

      cfg = self.config_manager.load()
      tracking_cfg = cfg["tracking"]
      tracking_cfg["detector_mode"] = mode

      if mode == "api":
        tracking_cfg["max_api_fps"] = 20.0
        tracking_cfg["imgsz"] = 640
      elif mode == "local":
        tracking_cfg["max_api_fps"] = 5.0
        tracking_cfg["imgsz"] = 480

      self.config_manager.save(cfg)
      self.config = cfg

      new_detector = build_person_detector(
        tracking_cfg=tracking_cfg,
        config_dir=self.config_dir,
      )
      with self._lock:
        self.detector = new_detector
        self.tracking_module.detector = new_detector
        self.tracking_module._track_memory.clear()

      print(f"[INFO] Detektor-Modus gewechselt: {mode}")
      return {"ok": True, "detector_mode": mode}

    def get_video_source(self) -> Dict[str, Any]:
      simulation_clip_id = self.simulation_registry.find_clip_id_by_path(str(getattr(self.video_input, "raw_source", "")))
      with self._simulation_lock:
        pending_count = len(self._simulation_pending_clip_ids)
        active_clip_id = self._simulation_active_clip_id
        active_kind = self._simulation_active_kind
        basic_clip_id = self._simulation_basic_clip_id
        pending_queue = self._simulation_queue_snapshot_locked()
      with self._video_input_lock:
        return {
          "mode": self.video_mode,
          "source": str(getattr(self.video_input, "raw_source", "")),
          "active_source": str(getattr(self.video_input, "active_source", getattr(self.video_input, "source", ""))),
          "opened": bool(self.video_input.capture and self.video_input.capture.isOpened()),
          "simulation_control_mode": self.simulation_control_mode,
          "simulation_idle_loop": bool(self.simulation_idle_loop),
          "simulation_clip_id": simulation_clip_id,
          "simulation_active_clip_id": active_clip_id,
          "simulation_active_kind": active_kind,
          "simulation_basic_clip_id": basic_clip_id,
          "simulation_pending_count": pending_count,
          "simulation_active_clip": self._simulation_clip_brief(active_clip_id),
          "simulation_basic_clip": self._simulation_clip_brief(basic_clip_id),
          "simulation_pending_queue": pending_queue,
        }

    def get_simulation_catalog(self) -> Dict[str, Any]:
      summary = self.simulation_registry.refresh()
      clips = []
      for clip in self.simulation_registry.list_clips():
        item = dict(clip)
        item["aliases_count"] = len(item.get("aliases", []))
        clips.append(item)
      return {"ok": True, "summary": summary, "clips": clips}

    def select_simulation_clip(self, payload: Dict[str, Any]) -> Dict[str, Any]:
      payload = dict(payload or {})
      action = str(payload.get("action", "play_now")).strip().lower() or "play_now"
      if action not in {"play_now", "enqueue", "play_next", "clear_queue"}:
        raise ValueError(f"action unbekannt: {action}")

      clip_id = str(payload.get("clip_id", "")).strip()
      clip = None
      if action != "clear_queue":
        if not clip_id:
          raise ValueError("clip_id fehlt")

        clip = self.simulation_registry.get_clip(clip_id)
        if not clip:
          self.simulation_registry.refresh()
          clip = self.simulation_registry.get_clip(clip_id)
        if not clip:
          raise ValueError(f"clip_id unbekannt: {clip_id}")

      cfg = self.config_manager.load()
      video_cfg = self._ensure_simulation_defaults(dict(cfg.get("video", {}) or {}))
      sim_cfg = dict(video_cfg.get("simulation", {}) or {})
      sim_cfg["control_mode"] = self._sanitize_simulation_control_mode(payload.get("control_mode", sim_cfg.get("control_mode", "remote_control")))
      sim_cfg["idle_loop"] = self._sanitize_simulation_idle_loop(payload.get("idle_loop", sim_cfg.get("idle_loop", self.simulation_idle_loop)))
      video_cfg["simulation"] = sim_cfg
      video_cfg["input_mode"] = "livefeed_simulation"

      basic_clip = self._resolve_basic_simulation_clip(video_cfg)
      if not basic_clip:
        basic_clip = clip
        sim_cfg["default_clip_id"] = clip.clip_id

      video_cfg["source"] = basic_clip.canonical_path
      cfg["video"] = video_cfg
      self.config_manager.save(cfg)

      self.config = cfg
      self.video_mode = "livefeed_simulation"
      self.simulation_control_mode = str(sim_cfg.get("control_mode", "remote_control"))
      self.simulation_idle_loop = self._sanitize_simulation_idle_loop(sim_cfg.get("idle_loop", self.simulation_idle_loop))

      with self._simulation_lock:
        self._simulation_basic_clip_id = basic_clip.clip_id
        if action == "clear_queue":
          self._simulation_pending_clip_ids.clear()
        elif action == "enqueue" and clip is not None:
          self._simulation_pending_clip_ids.append(clip.clip_id)
        elif action == "play_next" and clip is not None:
          self._simulation_pending_clip_ids.appendleft(clip.clip_id)
        elif action == "play_now" and clip is not None:
          self._simulation_pending_clip_ids.clear()

      opened = True
      if action == "play_now" and clip is not None:
        opened = self._switch_to_simulation_clip_id(clip.clip_id, kind="scheduled")
        if not opened:
          opened = self._switch_to_simulation_clip_id(basic_clip.clip_id, kind="basic")
      elif action in {"enqueue", "play_next", "clear_queue"}:
        with self._simulation_lock:
          active_clip_id = str(self._simulation_active_clip_id or "").strip()
        if not active_clip_id:
          opened = self._switch_to_simulation_clip_id(basic_clip.clip_id, kind="basic")

      with self._simulation_lock:
        pending_queue = self._simulation_queue_snapshot_locked()
        active_clip_id = self._simulation_active_clip_id
        active_kind = self._simulation_active_kind

      return {
        "ok": True,
        "action": action,
        "opened": bool(opened),
        "selected": (
          {
            "clip_id": clip.clip_id,
            "display_name": clip.display_name,
            "canonical_path": clip.canonical_path,
            "relative_path": clip.relative_path,
          }
          if clip is not None
          else None
        ),
        "next": {
          "clip_id": basic_clip.clip_id,
          "display_name": basic_clip.display_name,
        },
        "active": self._simulation_clip_brief(active_clip_id),
        "active_kind": active_kind,
        "simulation_idle_loop": bool(self.simulation_idle_loop),
        "queue": pending_queue,
      }

    def update_simulation_control(self, payload: Dict[str, Any]) -> Dict[str, Any]:
      payload = dict(payload or {})
      control_mode = self._sanitize_simulation_control_mode(
        payload.get("control_mode", self.simulation_control_mode)
      )
      idle_loop = self._sanitize_simulation_idle_loop(
        payload.get("idle_loop", self.simulation_idle_loop)
      )

      cfg = self.config_manager.load()
      video_cfg = self._ensure_simulation_defaults(dict(cfg.get("video", {}) or {}))
      sim_cfg = dict(video_cfg.get("simulation", {}) or {})
      sim_cfg["control_mode"] = control_mode
      sim_cfg["idle_loop"] = idle_loop
      video_cfg["simulation"] = sim_cfg
      cfg["video"] = video_cfg
      self.config_manager.save(cfg)

      self.config = cfg
      self.simulation_control_mode = control_mode
      self.simulation_idle_loop = idle_loop

      if self.video_mode == "livefeed_simulation":
        with self._simulation_lock:
          active_clip_id = str(self._simulation_active_clip_id or "").strip()
          active_kind = str(self._simulation_active_kind or "basic")
        if active_clip_id:
          self._switch_to_simulation_clip_id(active_clip_id, kind=active_kind)

      return {"ok": True, "video_source": self.get_video_source()}

    def update_video_source(self, payload: Dict[str, Any]) -> Dict[str, Any]:
      payload = dict(payload or {})
      mode = self._sanitize_video_mode(payload.get("mode", self.video_mode))
      source_value = payload.get("source", self.config.get("video", {}).get("source", "0"))
      source = str(source_value).strip() or str(self.config.get("video", {}).get("source", "0"))
      simulation_control_mode = self._sanitize_simulation_control_mode(
        payload.get(
          "simulation_control_mode",
          self.config.get("video", {}).get("simulation", {}).get("control_mode", self.simulation_control_mode),
        )
      )
      simulation_idle_loop = self._sanitize_simulation_idle_loop(
        payload.get(
          "simulation_idle_loop",
          self.config.get("video", {}).get("simulation", {}).get("idle_loop", self.simulation_idle_loop),
        )
      )

      cfg = self.config_manager.load()
      video_cfg = self._ensure_simulation_defaults(dict(cfg.get("video", {}) or {}))
      video_cfg["input_mode"] = mode
      video_cfg["source"] = source
      video_cfg.setdefault("simulation", {})["control_mode"] = simulation_control_mode
      video_cfg.setdefault("simulation", {})["idle_loop"] = simulation_idle_loop
      cfg["video"] = video_cfg
      self.config_manager.save(cfg)
      self.config = cfg
      self.video_mode = mode
      self.simulation_control_mode = simulation_control_mode
      self.simulation_idle_loop = simulation_idle_loop
      if self.video_mode == "livefeed_simulation":
        self._initialize_simulation_state(video_cfg)

      replacement_input = self._create_video_input(video_cfg)

      with self._video_input_lock:
        previous_input = self.video_input
        self.video_input = replacement_input
        try:
          previous_input.release()
        except Exception:
          pass
        opened = self.video_input.open()

      return {
        "ok": True,
        "mode": self.video_mode,
        "source": str(self.video_input.raw_source),
        "active_source": str(getattr(self.video_input, "active_source", self.video_input.source)),
        "opened": bool(opened),
        "simulation_control_mode": self.simulation_control_mode,
        "simulation_idle_loop": bool(self.simulation_idle_loop),
      }

    def get_zone(self) -> Dict[str, Any]:
        return self.zone_config.to_dict()

    def update_zone(self, zone_payload: Dict[str, Any]):
        updated = EntranceZoneConfig.from_dict(zone_payload)
        self.zone_config = updated
        self.entry_analysis.set_zone_config(updated)
        self.config_manager.update_zone(updated.to_dict())

    @staticmethod
    def _normalize_analysis_roi(payload: Dict[str, Any]) -> Dict[str, Any]:
      defaults = {
        "enabled": False,
        "mode": "rect",
        "x_min": 0.0,
        "y_min": 0.0,
        "x_max": 1.0,
        "y_max": 1.0,
        "polygon_points": [],
      }
      merged = {**defaults, **(payload or {})}
      mode = "polygon" if str(merged.get("mode", "rect")).lower() == "polygon" else "rect"
      x_min = max(0.0, min(1.0, float(merged.get("x_min", 0.0))))
      y_min = max(0.0, min(1.0, float(merged.get("y_min", 0.0))))
      x_max = max(0.0, min(1.0, float(merged.get("x_max", 1.0))))
      y_max = max(0.0, min(1.0, float(merged.get("y_max", 1.0))))

      polygon_points = []
      for pt in merged.get("polygon_points", []) or []:
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
          continue
        px = max(0.0, min(1.0, float(pt[0])))
        py = max(0.0, min(1.0, float(pt[1])))
        polygon_points.append([px, py])

      if x_max <= x_min:
        x_max = min(1.0, x_min + 0.01)
      if y_max <= y_min:
        y_max = min(1.0, y_min + 0.01)

      return {
        "enabled": bool(merged.get("enabled", False)),
        "mode": mode,
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "polygon_points": polygon_points,
      }

    def _crop_to_analysis_roi(self, frame):
      roi = self.analysis_roi
      if not roi.get("enabled", False):
        return frame, (0, 0)

      mode = str(roi.get("mode", "rect")).lower()
      if mode == "polygon":
        polygon_points_raw = roi.get("polygon_points", [])
        polygon_points = polygon_points_raw if isinstance(polygon_points_raw, list) else []
        if len(polygon_points) >= 3:
          frame_h, frame_w = frame.shape[:2]
          points_px = []
          for point in polygon_points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
              continue
            px = int(float(point[0]) * frame_w)
            py = int(float(point[1]) * frame_h)
            points_px.append([px, py])
          if len(points_px) < 3:
            return frame, (0, 0)

          pts = np.array(points_px, dtype=np.int32)
          x, y, w, h = cv2.boundingRect(pts)
          if w >= 8 and h >= 8:
            crop = frame[y : y + h, x : x + w]
            shifted_pts = pts - np.array([x, y], dtype=np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [shifted_pts], 255)
            masked_crop = cv2.bitwise_and(crop, crop, mask=mask)
            return masked_crop, (x, y)

      frame_h, frame_w = frame.shape[:2]
      x1 = max(0, min(frame_w - 1, int(roi["x_min"] * frame_w)))
      y1 = max(0, min(frame_h - 1, int(roi["y_min"] * frame_h)))
      x2 = max(1, min(frame_w, int(roi["x_max"] * frame_w)))
      y2 = max(1, min(frame_h, int(roi["y_max"] * frame_h)))

      if x2 - x1 < 8 or y2 - y1 < 8:
        return frame, (0, 0)

      return frame[y1:y2, x1:x2], (x1, y1)

    @staticmethod
    def _remap_tracks_to_full_frame(tracks, roi_offset):
      offset_x, offset_y = roi_offset
      remapped = []

      for track in tracks:
        item = dict(track)
        bbox = item.get("bbox")
        center = item.get("center")
        trail = item.get("trail")

        if bbox and len(bbox) == 4:
          item["bbox"] = [
            float(bbox[0]) + offset_x,
            float(bbox[1]) + offset_y,
            float(bbox[2]) + offset_x,
            float(bbox[3]) + offset_y,
          ]
        if center and len(center) == 2:
          item["center"] = (float(center[0]) + offset_x, float(center[1]) + offset_y)
        if isinstance(trail, list):
          item["trail"] = [
            (float(pt[0]) + offset_x, float(pt[1]) + offset_y)
            for pt in trail
            if isinstance(pt, (list, tuple)) and len(pt) == 2
          ]

        remapped.append(item)

      return remapped

    def get_analysis_roi(self) -> Dict[str, Any]:
      return dict(self.analysis_roi)

    def update_analysis_roi(self, roi_payload: Dict[str, Any]):
      roi = self._normalize_analysis_roi(roi_payload)
      self.analysis_roi = roi

      config = self.config_manager.load()
      config.setdefault("tracking", {})["analysis_roi"] = roi
      self.config_manager.save(config)

    def set_analysis_enabled(self, enabled: bool) -> Dict[str, Any]:
      with self._lock:
        self.analysis_enabled = bool(enabled)
        if not self.analysis_enabled:
          self._analysis_queue.clear()
          self._last_tracks_cache = []
          # Ensure packetizer switches immediately to capture frames.
          self._latest_visual_jpeg = None
          self._latest_visual_ts = 0.0
      return {"ok": True, "analysis_enabled": bool(self.analysis_enabled)}

    def get_analysis_control(self) -> Dict[str, Any]:
      with self._lock:
        return {"ok": True, "analysis_enabled": bool(self.analysis_enabled)}

    def get_profile_simulation_control(self) -> Dict[str, Any]:
      return {"ok": True, "profile_simulation": self.profile_simulation.get_status()}

    def update_profile_simulation_control(self, payload: Dict[str, Any]) -> Dict[str, Any]:
      payload = dict(payload or {})
      raw_cfg = payload.get("profile_simulation", payload)
      if not isinstance(raw_cfg, dict):
        raw_cfg = {}
      cfg_update = dict(raw_cfg)

      if "excel_path" in cfg_update:
        excel_path = str(cfg_update.get("excel_path", "")).strip()
        if excel_path and not os.path.isabs(excel_path):
          excel_path = self._resolve_config_relative_path(excel_path)
        cfg_update["excel_path"] = excel_path

      self.profile_simulation.update_settings(cfg_update)

      config = self.config_manager.load()
      dashboard_cfg = dict(config.get("dashboard", {}) or {})
      stored_cfg = dict(dashboard_cfg.get("profile_simulation", {}) or {})
      allowed_keys = {
        "enabled",
        "excel_path",
        "tick_seconds",
        "profile_blend",
        "noise_sigma_scale",
        "max_step_per_tick",
        "rollback_minutes",
      }
      for key in allowed_keys:
        if key in cfg_update:
          stored_cfg[key] = cfg_update[key]
      dashboard_cfg["profile_simulation"] = stored_cfg
      config["dashboard"] = dashboard_cfg
      self.config_manager.save(config)
      self.config = config

      return {"ok": True, "profile_simulation": self.profile_simulation.get_status()}


def create_app(config_path: str) -> Flask:
    app = Flask(__name__)
    engine = TrackingEngine(config_path=config_path)
    engine.start()

    @app.route("/")
    def index():
        return render_template_string(
            HTML_PAGE,
        dash_enabled=engine.dash_streamer.enabled,
        )

    @app.route("/dash/<path:filename>")
    def dash_files(filename: str):
      return send_from_directory(engine.dash_streamer.output_dir, filename, conditional=False)

    @app.route("/dash")
    def dash_root():
      return send_from_directory(engine.dash_streamer.output_dir, "stream.mpd", conditional=False)

    @app.route("/api/state")
    def api_state():
        return jsonify(engine.get_state())

    @app.route("/api/zone", methods=["GET", "POST"])
    def api_zone():
        if request.method == "GET":
            return jsonify(engine.get_zone())

        payload = request.get_json(silent=True) or {}
        try:
            engine.update_zone(payload)
            return jsonify({"ok": True, "zone": engine.get_zone()})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.route("/api/tracking-roi", methods=["GET", "POST"])
    def api_tracking_roi():
        if request.method == "GET":
            return jsonify({"ok": True, "analysis_roi": engine.get_analysis_roi()})

        payload = request.get_json(silent=True) or {}
        roi_payload = payload.get("analysis_roi", payload)
        try:
            engine.update_analysis_roi(roi_payload)
            return jsonify({"ok": True, "analysis_roi": engine.get_analysis_roi()})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.route("/api/analysis-control", methods=["GET", "POST"])
    def api_analysis_control():
        if request.method == "GET":
            return jsonify(engine.get_analysis_control())

        payload = request.get_json(silent=True) or {}
        try:
            enabled = bool(payload.get("enabled", True))
            return jsonify(engine.set_analysis_enabled(enabled))
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.route("/api/video-source", methods=["GET", "POST"])
    def api_video_source():
        if request.method == "GET":
            return jsonify(engine.get_video_source())

        payload = request.get_json(silent=True) or {}
        try:
            return jsonify(engine.update_video_source(payload))
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.route("/api/simulation/catalog", methods=["GET"])
    def api_simulation_catalog():
      try:
        return jsonify(engine.get_simulation_catalog())
      except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    @app.route("/api/simulation/select", methods=["POST"])
    def api_simulation_select():
      payload = request.get_json(silent=True) or {}
      try:
        return jsonify(engine.select_simulation_clip(payload))
      except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    @app.route("/api/simulation/control", methods=["POST"])
    def api_simulation_control():
      payload = request.get_json(silent=True) or {}
      try:
        return jsonify(engine.update_simulation_control(payload))
      except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    @app.route("/api/profile-simulation", methods=["GET", "POST"])
    def api_profile_simulation():
      if request.method == "GET":
        return jsonify(engine.get_profile_simulation_control())

      payload = request.get_json(silent=True) or {}
      try:
        return jsonify(engine.update_profile_simulation_control(payload))
      except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    @app.route("/api/detector/mode", methods=["GET", "POST"])
    def api_detector_mode():
      if request.method == "GET":
        mode = str(engine.config.get("tracking", {}).get("detector_mode", "hybrid"))
        return jsonify({"ok": True, "detector_mode": mode})
      payload = request.get_json(silent=True) or {}
      try:
        return jsonify(engine.set_detector_mode(payload))
      except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    @app.route("/health")
    def health():
        return jsonify({"ok": True})

    return app


def main():
    parser = argparse.ArgumentParser(description="Sitcheck live web dashboard")
    parser.add_argument("--config", default=os.path.join(WORKSPACE_ROOT, "bildauswertung", "config.yaml"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    app = create_app(config_path=args.config)
    app.run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
