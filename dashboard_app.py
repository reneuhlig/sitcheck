#!/usr/bin/env python3
import argparse
from collections import deque
import os
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request, send_from_directory

from ConfigManager import ConfigManager
from OccupancyStateModule import OccupancyStateModule
from TrajectoryEntryAnalysisModule import EntranceZoneConfig, TrajectoryEntryAnalysisModule
from UltralyticsPersonDetector import UltralyticsPersonDetector
from VideoInputModule import VideoInputModule
from VisualizationOutputModule import VisualizationOutputModule
from YOLOTrackingModule import YOLOTrackingModule


HTML_PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Sitcheck Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; }
    .wrap { max-width: 1360px; margin: 0 auto; padding: 16px; }
    .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 16px; }
    .panel { background: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 12px; }
    .title { margin: 0 0 10px; font-size: 16px; }
    .feed-wrap { position: relative; width: 100%; aspect-ratio: 16 / 9; background: #000; overflow: hidden; border-radius: 8px; }
    #dashFeed { width: 100%; height: 100%; object-fit: contain; display: block; }
    #dashFeed { background: #000; }
    #overlay { position: absolute; inset: 0; pointer-events: auto; }
    .row { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0; }
    .stat { background: #0b1220; border: 1px solid #1f2937; border-radius: 8px; padding: 8px; flex: 1 1 120px; }
    button, select { background: #1f2937; color: #e5e7eb; border: 1px solid #374151; border-radius: 6px; padding: 8px 10px; cursor: pointer; }
    button:hover, select:hover { background: #374151; }
    .muted { color: #9ca3af; font-size: 13px; }
    .ok { color: #22c55e; }
    .warn { color: #f59e0b; }
  </style>
</head>
<body>
  <div class="wrap">
    <h2 style="margin-top: 0;">Sitcheck YOLO26 Dashboard</h2>
    <div class="grid">
      <div class="panel">
        <h3 class="title">Live Feed + Tracking</h3>
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
          <div class="stat"><b>Occupancy</b><div id="occupancy">-</div></div>
          <div class="stat"><b>Entries</b><div id="entries">-</div></div>
          <div class="stat"><b>Exits</b><div id="exits">-</div></div>
        </div>
        <div class="row">
          <div class="stat"><b>Tracks</b><div id="tracks">-</div></div>
          <div class="stat"><b>FPS</b><div id="fps">-</div></div>
          <div class="stat"><b>Infer FPS</b><div id="infer_fps">-</div></div>
          <div class="stat"><b>Frame Ev.</b><div id="events">-</div></div>
        </div>
        <div id="status" class="muted">Loading...</div>

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
    const entriesEl = document.getElementById('entries');
    const exitsEl = document.getElementById('exits');
    const tracksEl = document.getElementById('tracks');
    const fpsEl = document.getElementById('fps');
    const inferFpsEl = document.getElementById('infer_fps');
    const eventsEl = document.getElementById('events');
    const statusEl = document.getElementById('status');
    const streamStatusEl = document.getElementById('stream_status');
    const activePolySel = document.getElementById('activePoly');
    const roiEnabledEl = document.getElementById('roi_enabled');
    const roiModeEl = document.getElementById('roi_mode');
    const roiEditModeEl = document.getElementById('roi_edit_mode');
    const roiXMinEl = document.getElementById('roi_xmin');
    const roiYMinEl = document.getElementById('roi_ymin');
    const roiXMaxEl = document.getElementById('roi_xmax');
    const roiYMaxEl = document.getElementById('roi_ymax');
    const DASH_ENABLED = {{ dash_enabled|tojson }};

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
    let dashController = null;

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
      const dashUrl = `/dash/stream.mpd?_t=${Date.now()}`;
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

    function drawZone() {
      syncCanvasSize();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.lineWidth = 2;

      if (zone.mode === 'line') {
        const p1 = [zone.line.p1[0] * canvas.width, zone.line.p1[1] * canvas.height];
        const p2 = [zone.line.p2[0] * canvas.width, zone.line.p2[1] * canvas.height];
        ctx.strokeStyle = '#00ffff';
        ctx.beginPath();
        ctx.moveTo(p1[0], p1[1]);
        ctx.lineTo(p2[0], p2[1]);
        ctx.stroke();
        ctx.fillStyle = '#ffff00';
        ctx.beginPath(); ctx.arc(p1[0], p1[1], 5, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(p2[0], p2[1], 5, 0, Math.PI * 2); ctx.fill();
      } else if (zone.mode === 'polygon') {
        const pts = zone.polygon.points.map(p => [p[0] * canvas.width, p[1] * canvas.height]);
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
        const entryPts = zone.entry_polygon.points.map(p => [p[0] * canvas.width, p[1] * canvas.height]);
        const exitPts = zone.exit_polygon.points.map(p => [p[0] * canvas.width, p[1] * canvas.height]);

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
          const pts = (analysisRoi.polygon_points || []).map(p => [p[0] * canvas.width, p[1] * canvas.height]);
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
          const rx1 = analysisRoi.x_min * canvas.width;
          const ry1 = analysisRoi.y_min * canvas.height;
          const rx2 = analysisRoi.x_max * canvas.width;
          const ry2 = analysisRoi.y_max * canvas.height;
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
      const res = await fetch('/api/tracking-roi');
      const payload = await res.json();
      setRoiForm(payload.analysis_roi || payload);
      drawZone();
    }

    async function saveRoi() {
      const roi = getRoiFromForm();
      const res = await fetch('/api/tracking-roi', {
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
      const x = (ev.clientX - rect.left) / rect.width;
      const y = (ev.clientY - rect.top) / rect.height;
      return [Math.max(0, Math.min(1, x)), Math.max(0, Math.min(1, y))];
    }

    async function saveZone() {
      zone.mode = modeSel.value;
      zone.line.entry_direction = directionSel.value;
      const res = await fetch('/api/zone', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(zone)
      });
      const data = await res.json();
      statusEl.textContent = data.ok ? 'Zone saved and applied.' : ('Error: ' + (data.error || 'unknown'));
      statusEl.className = data.ok ? 'ok' : 'warn';
    }

    async function reloadZone() {
      const res = await fetch('/api/zone');
      zone = await res.json();
      modeSel.value = zone.mode;
      directionSel.value = zone.line.entry_direction;
      lineStage = 0;
      drawZone();
    }

    async function pollState() {
      try {
        const res = await fetch('/api/state');
        const st = await res.json();
        occupancyEl.textContent = st.occupancy;
        entriesEl.textContent = st.entries_total;
        exitsEl.textContent = st.exits_total;
        tracksEl.textContent = st.tracks;
        fpsEl.textContent = st.fps;
        inferFpsEl.textContent = st.inference_fps;
        eventsEl.textContent = `+${st.entries_frame} / -${st.exits_frame}`;
      } catch (e) {}
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

    reloadZone();
    loadRoi();
    initDashPlayer();
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
        self.config_path = config_path
        self.config_manager = ConfigManager(config_path=config_path)
        self.config = self.config_manager.load()

        tracking_cfg = self.config["tracking"]

        self.detector = UltralyticsPersonDetector(
            model_path=tracking_cfg["model_path"],
            confidence_threshold=float(tracking_cfg["confidence_threshold"]),
            device=str(tracking_cfg.get("device", "cpu")),
        )

        self.video_input = VideoInputModule(
            source=str(self.config["video"]["source"]),
            reconnect_delay=float(self.config["video"]["reconnect_delay"]),
            max_retries=int(self.config["video"]["max_retries"]),
        )

        self.tracking_module = YOLOTrackingModule(
            detector=self.detector,
            tracker_config=str(tracking_cfg["tracker"]),
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
        self.entry_analysis = TrajectoryEntryAnalysisModule(zone_config=self.zone_config)
        self.occupancy_state = OccupancyStateModule(db=None)
        self.visualizer = VisualizationOutputModule(show_window=False, enable_zone_editor=False)

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
        self.dash_output_dir = str(dash_cfg.get("output_dir", "dash"))
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
        self.last_entries = 0
        self.last_exits = 0
        self.last_fps = 0.0
        self.last_inference_fps = 0.0
        self.last_track_ok = True
        self.last_track_error = ""
        self._fps_ema = 0.0
        self.analysis_skipped_frames = 0

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

    def start(self):
      if self._running:
        return
      self.video_input.open()
      self.dash_streamer.start()
      self._running = True
      self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
      self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
      self._packetizer_thread = threading.Thread(target=self._packetizer_loop, daemon=True)
      self._capture_thread.start()
      self._inference_thread.start()
      self._packetizer_thread.start()

    def stop(self):
      self._running = False
      if self._capture_thread:
        self._capture_thread.join(timeout=2)
      if self._inference_thread:
        self._inference_thread.join(timeout=2)
      if self._packetizer_thread:
        self._packetizer_thread.join(timeout=2)
      self.dash_streamer.stop()
      self.video_input.release()

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
        ok, frame = self.video_input.read()
        if not ok or frame is None:
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
          if len(self._analysis_queue) >= self.analysis_queue_frames:
            try:
              self._analysis_queue.popleft()
              self.analysis_skipped_frames += 1
            except IndexError:
              pass

          self._capture_frame_idx += 1
          frame_id = self._capture_frame_idx
          self._analysis_queue.append((frame_id, frame))

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

        events = self.entry_analysis.update(tracks=tracks, frame_shape=frame.shape) if run_tracking_now else []
        frame_entries = 0
        frame_exits = 0
        for event in events:
          if self.occupancy_state.handle_event(event):
            if str(event.get("type", "entry")).lower() == "entry":
              self.entries_total += 1
              frame_entries += 1
            elif str(event.get("type", "entry")).lower() == "exit":
              self.exits_total += 1
              frame_exits += 1

        now_visual = time.monotonic()
        should_update_visual = run_tracking_now
        if not should_update_visual and self.visual_update_interval > 0:
          should_update_visual = (now_visual - self._last_visual_update_ts) >= self.visual_update_interval

        if should_update_visual:
          output = self.visualizer.draw(
            frame=frame,
            tracks=tracks,
            zone_config=self.zone_config,
            occupancy=self.occupancy_state.occupancy,
            entries_total=self.entries_total,
            exits_total=self.exits_total,
            events_in_frame={"entry": frame_entries, "exit": frame_exits},
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

          if visual_valid and raw_valid:
            encoded_frame = self._latest_visual_jpeg
          elif visual_valid:
            encoded_frame = self._latest_visual_jpeg
          elif self._latest_visual_jpeg is not None:
            encoded_frame = self._latest_visual_jpeg
          elif raw_valid:
            encoded_frame = self._latest_raw_jpeg
          elif self._latest_raw_jpeg is not None:
            encoded_frame = self._latest_raw_jpeg

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
            return {
                "occupancy": self.occupancy_state.occupancy,
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
                "zone": self.zone_config.to_dict(),
                "analysis_roi": self.analysis_roi,
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

    @app.route("/health")
    def health():
        return jsonify({"ok": True})

    return app


def main():
    parser = argparse.ArgumentParser(description="Sitcheck live web dashboard")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    app = create_app(config_path=args.config)
    app.run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
