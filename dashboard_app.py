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
    #hlsFeed { width: 100%; height: 100%; object-fit: contain; display: block; }
    #hlsFeed { background: #000; }
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
          <video id="hlsFeed" muted autoplay playsinline></video>
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
      </div>
    </div>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
  <script>
    const hlsFeed = document.getElementById('hlsFeed');
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
    const HLS_ENABLED = {{ hls_enabled|tojson }};

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
    let hlsController = null;

    function initHlsPlayer() {
      if (!HLS_ENABLED) {
        streamStatusEl.textContent = 'HLS ist deaktiviert';
        return;
      }

      streamStatusEl.textContent = 'HLS wird initialisiert...';

      const hlsUrl = `/hls/stream.m3u8?_t=${Date.now()}`;
      if (window.Hls && window.Hls.isSupported()) {
        hlsController = new window.Hls({
          lowLatencyMode: true,
          liveSyncDurationCount: 3,
          liveMaxLatencyDurationCount: 10,
          backBufferLength: 30,
        });
        hlsController.loadSource(hlsUrl);
        hlsController.attachMedia(hlsFeed);
        hlsController.on(window.Hls.Events.MANIFEST_PARSED, () => {
          streamStatusEl.textContent = 'HLS aktiv';
          hlsFeed.play().catch(() => {});
        });
        hlsController.on(window.Hls.Events.ERROR, (_ev, data) => {
          if (data && data.fatal) {
            streamStatusEl.textContent = 'HLS Fehler: Stream neu laden';
          }
        });
      } else if (hlsFeed.canPlayType('application/vnd.apple.mpegurl')) {
        hlsFeed.src = hlsUrl;
        hlsFeed.addEventListener('loadedmetadata', () => {
          streamStatusEl.textContent = 'HLS aktiv';
          hlsFeed.play().catch(() => {});
        }, { once: true });
      } else {
        streamStatusEl.textContent = 'Browser unterstützt HLS nicht';
      }
    }

    function syncCanvasSize() {
      const rect = hlsFeed.getBoundingClientRect();
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
      lineStageEl.textContent = String(lineStage);
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
    modeSel.addEventListener('change', drawZone);
    directionSel.addEventListener('change', drawZone);
    activePolySel.addEventListener('change', drawZone);

    window.addEventListener('resize', drawZone);

    reloadZone();
    initHlsPlayer();
    setInterval(pollState, 1000);
    setInterval(drawZone, 1000);
  </script>
</body>
</html>
"""


class HLSStreamer:
  def __init__(self, output_dir: str, fps: float, segment_time: float = 1.0, list_size: int = 12):
    self.output_dir = output_dir
    self.fps = max(1.0, float(fps))
    self.segment_time = max(0.5, float(segment_time))
    self.list_size = max(3, int(list_size))
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
      if name.endswith(".ts") or name.endswith(".m3u8"):
        try:
          os.remove(os.path.join(self.output_dir, name))
        except OSError:
          pass

    segment_pattern = os.path.join(self.output_dir, "seg_%06d.ts")
    playlist_path = os.path.join(self.output_dir, "stream.m3u8")
    gop = max(10, int(round(self.fps * 2.0)))

    command = [
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
      "-c:v",
      "libx264",
      "-preset",
      "veryfast",
      "-tune",
      "zerolatency",
      "-pix_fmt",
      "yuv420p",
      "-g",
      str(gop),
      "-keyint_min",
      str(gop),
      "-sc_threshold",
      "0",
      "-f",
      "hls",
      "-hls_time",
      str(self.segment_time),
      "-hls_list_size",
      str(self.list_size),
      "-hls_flags",
      "delete_segments+append_list+independent_segments+omit_endlist",
      "-hls_segment_filename",
      segment_pattern,
      playlist_path,
    ]

    self._process = subprocess.Popen(
      command,
      stdin=subprocess.PIPE,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
    )

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
          preprocess_enabled=bool(self.config.get("preprocess", {}).get("enabled", False)),
          preprocess_upscale=float(self.config.get("preprocess", {}).get("upscale", 1.0)),
          preprocess_clahe_clip=float(self.config.get("preprocess", {}).get("clahe_clip", 2.0)),
          preprocess_denoise=bool(self.config.get("preprocess", {}).get("denoise", False)),
        )
        self.process_every_n_frames = max(1, int(tracking_cfg.get("process_every_n_frames", 1)))

        self.zone_config = EntranceZoneConfig.from_dict(self.config["zone"])
        self.entry_analysis = TrajectoryEntryAnalysisModule(zone_config=self.zone_config)
        self.occupancy_state = OccupancyStateModule(db=None)
        self.visualizer = VisualizationOutputModule(show_window=False, enable_zone_editor=False)

        dashboard_cfg = self.config.get("dashboard", {})
        self.stream_fps = float(dashboard_cfg.get("stream_fps", 25.0))
        self.jpeg_quality = max(40, min(95, int(dashboard_cfg.get("jpeg_quality", 80))))
        self.stream_max_width = max(320, int(dashboard_cfg.get("stream_max_width", 960)))
        hls_cfg = dashboard_cfg.get("hls", {})
        self.hls_enabled = bool(hls_cfg.get("enabled", True))
        self.hls_output_dir = str(hls_cfg.get("output_dir", "hls"))
        self.hls_segment_time = float(hls_cfg.get("segment_time", 1.0))
        self.hls_list_size = int(hls_cfg.get("list_size", 12))
        self.analysis_queue_frames = max(8, int(dashboard_cfg.get("analysis_queue_frames", 64)))
        self.analysis_skip_threshold_frames = max(0, int(dashboard_cfg.get("analysis_skip_threshold_frames", 0)))
        self.hls_streamer = HLSStreamer(
          output_dir=self.hls_output_dir,
          fps=self.stream_fps,
          segment_time=self.hls_segment_time,
          list_size=self.hls_list_size,
        )
        if not self.hls_enabled:
          self.hls_streamer._enabled = False

        self.entries_total = 0
        self.exits_total = 0
        self.last_tracks = 0
        self.last_entries = 0
        self.last_exits = 0
        self.last_fps = 0.0
        self.last_inference_fps = 0.0
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
        self._hls_frame_counter = 0

    def start(self):
      if self._running:
        return
      self.video_input.open()
      self.hls_streamer.start()
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
      self.hls_streamer.stop()
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
          1,
        ],
      )

    def _capture_loop(self):
      while self._running:
        frame = None
        ok, frame = self.video_input.read()
        if not ok or frame is None:
          time.sleep(0.02)
          continue

        enqueued = False
        while self._running and not enqueued:
          with self._lock:
            if len(self._analysis_queue) < self.analysis_queue_frames:
              self._capture_frame_idx += 1
              frame_id = self._capture_frame_idx
              self._analysis_queue.append((frame_id, frame))
              enqueued = True
          if not enqueued:
            time.sleep(0.002)

        ok_jpeg, encoded = self._encode_stream_jpeg(frame)
        if ok_jpeg:
          with self._packet_cv:
            self._latest_raw_jpeg = encoded.tobytes()
            self._packet_cv.notify_all()

    def _inference_loop(self):
      while self._running:
        t0 = time.time()

        with self._lock:
          if not self._analysis_queue:
            frame_tuple = None
          else:
            if self.analysis_skip_threshold_frames > 0:
              while len(self._analysis_queue) > self.analysis_skip_threshold_frames:
                self._analysis_queue.popleft()
                self.analysis_skipped_frames += 1
            frame_tuple = self._analysis_queue.popleft() if self._analysis_queue else None

        if frame_tuple is None:
          time.sleep(0.005)
          continue

        frame_id, frame = frame_tuple
        run_tracking_now = (frame_id % self.process_every_n_frames) == 0

        if run_tracking_now:
          infer_t0 = time.time()
          tracks = self.tracking_module.track(frame)
          self._last_tracks_cache = tracks
          infer_elapsed = max(1e-6, time.time() - infer_t0)
          self.last_inference_fps = round(1.0 / infer_elapsed, 2)
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

        output = self.visualizer.draw(
          frame=frame,
          tracks=tracks,
          zone_config=self.zone_config,
          occupancy=self.occupancy_state.occupancy,
          entries_total=self.entries_total,
          exits_total=self.exits_total,
          events_in_frame={"entry": frame_entries, "exit": frame_exits},
        )

        ok_jpeg, encoded = self._encode_stream_jpeg(output)
        if ok_jpeg:
          with self._packet_cv:
            self._latest_visual_jpeg = encoded.tobytes()
            self.last_tracks = len(tracks)
            self.last_entries = frame_entries
            self.last_exits = frame_exits
            elapsed = max(1e-6, time.time() - t0)
            self.last_fps = round(1.0 / elapsed, 2)
            self._packet_cv.notify_all()

    def _packetizer_loop(self):
      packet_interval = 1.0 / max(1.0, self.stream_fps)
      next_deadline = time.monotonic()

      while self._running:
        sleep_for = next_deadline - time.monotonic()
        if sleep_for > 0:
          time.sleep(sleep_for)

        with self._packet_cv:
          frame = self._latest_visual_jpeg or self._latest_raw_jpeg
          if frame is not None:
            self._hls_frame_counter += 1
            self.hls_streamer.write_jpeg(frame)
            if self._hls_frame_counter % max(1, int(self.stream_fps)) == 0:
              self.hls_streamer.write_preview(frame)
            self._packet_cv.notify_all()

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
                "hls_enabled": self.hls_streamer.enabled,
                "hls_output_dir": self.hls_output_dir,
                "zone": self.zone_config.to_dict(),
            }

    def get_zone(self) -> Dict[str, Any]:
        return self.zone_config.to_dict()

    def update_zone(self, zone_payload: Dict[str, Any]):
        updated = EntranceZoneConfig.from_dict(zone_payload)
        self.zone_config = updated
        self.entry_analysis.set_zone_config(updated)
        self.config_manager.update_zone(updated.to_dict())


def create_app(config_path: str) -> Flask:
    app = Flask(__name__)
    engine = TrackingEngine(config_path=config_path)
    engine.start()

    @app.route("/")
    def index():
        return render_template_string(
            HTML_PAGE,
        hls_enabled=engine.hls_streamer.enabled,
        )

    @app.route("/hls/<path:filename>")
    def hls_files(filename: str):
      return send_from_directory(engine.hls_streamer.output_dir, filename, conditional=False)

    @app.route("/hls")
    def hls_root():
      return send_from_directory(engine.hls_streamer.output_dir, "stream.m3u8", conditional=False)

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
