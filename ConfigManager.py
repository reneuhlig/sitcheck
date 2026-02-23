from __future__ import annotations

import copy
import os
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "video": {
        "source": "https://www.youtube.com/watch?v=8JCk5M_xrBs",
        "reconnect_delay": 1.0,
        "max_retries": 0,
    },
    "tracking": {
        "model_path": "yolo26n.pt",
        "device": "cpu",
        "tracker": "bytetrack.yaml",
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "imgsz": 512,
        "process_every_n_frames": 2,
    },
    "preprocess": {
        "enabled": False,
        "upscale": 1.0,
        "clahe_clip": 2.0,
        "denoise": False,
    },
    "dashboard": {
        "stream_fps": 25,
        "capture_buffer_size": 24,
        "model_buffer_size": 6,
        "model_latency_frames": 2,
        "render_buffer_size": 12,
    },
    "zone": {
        "mode": "dual_polygon",
        "line": {
            "p1": [0.35, 0.65],
            "p2": [0.65, 0.65],
            "entry_direction": "negative_to_positive",
        },
        "polygon": {
            "points": [[0.35, 0.55], [0.65, 0.55], [0.75, 0.85], [0.25, 0.85]],
        },
        "entry_polygon": {
            "points": [[0.30, 0.52], [0.55, 0.52], [0.60, 0.88], [0.25, 0.88]],
        },
        "exit_polygon": {
            "points": [[0.60, 0.52], [0.82, 0.52], [0.88, 0.88], [0.64, 0.88]],
        },
        "min_crossing_displacement_px": 20.0,
        "min_track_points": 4,
        "min_event_cooldown_frames": 8,
    },
    "ui": {
        "show_window": True,
        "enable_zone_editor": True,
        "window_name": "Library Entry Tracking",
    },
    "database": {
        "enabled": False,
        "host": "localhost",
        "user": "aiuser",
        "password": "DHBW1234!?",
        "database": "ai_detection",
        "port": 5432,
    },
}


class ConfigManager:
    """Lädt/speichert YAML-Konfiguration und überschreibt per ENV."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path

    def load(self) -> Dict[str, Any]:
        config = copy.deepcopy(DEFAULT_CONFIG)

        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            self._deep_update(config, loaded)

        self._apply_env_overrides(config)
        return config

    def save(self, config: Dict[str, Any]):
        with open(self.config_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    def update_zone(self, zone_dict: Dict[str, Any]):
        config = self.load()
        config["zone"] = zone_dict
        self.save(config)

    def _apply_env_overrides(self, config: Dict[str, Any]):
        env_map = {
            "SITCHECK_VIDEO_SOURCE": ("video", "source", str),
            "SITCHECK_YOLO_MODEL": ("tracking", "model_path", str),
            "SITCHECK_DEVICE": ("tracking", "device", str),
            "SITCHECK_TRACKER": ("tracking", "tracker", str),
            "SITCHECK_CONFIDENCE": ("tracking", "confidence_threshold", float),
            "SITCHECK_IOU": ("tracking", "iou_threshold", float),
            "SITCHECK_IMGSZ": ("tracking", "imgsz", int),
            "SITCHECK_PROCESS_EVERY_N_FRAMES": ("tracking", "process_every_n_frames", int),
            "SITCHECK_PREPROCESS_ENABLED": ("preprocess", "enabled", self._parse_bool),
            "SITCHECK_PREPROCESS_UPSCALE": ("preprocess", "upscale", float),
            "SITCHECK_PREPROCESS_CLAHE_CLIP": ("preprocess", "clahe_clip", float),
            "SITCHECK_PREPROCESS_DENOISE": ("preprocess", "denoise", self._parse_bool),
            "SITCHECK_DASH_STREAM_FPS": ("dashboard", "stream_fps", int),
            "SITCHECK_DASH_CAPTURE_BUFFER": ("dashboard", "capture_buffer_size", int),
            "SITCHECK_DASH_MODEL_BUFFER": ("dashboard", "model_buffer_size", int),
            "SITCHECK_DASH_MODEL_LATENCY": ("dashboard", "model_latency_frames", int),
            "SITCHECK_DASH_RENDER_BUFFER": ("dashboard", "render_buffer_size", int),
            "SITCHECK_SHOW_WINDOW": ("ui", "show_window", self._parse_bool),
            "SITCHECK_ZONE_EDITOR": ("ui", "enable_zone_editor", self._parse_bool),
            "SITCHECK_DB_ENABLED": ("database", "enabled", self._parse_bool),
            "SITCHECK_DB_HOST": ("database", "host", str),
            "SITCHECK_DB_USER": ("database", "user", str),
            "SITCHECK_DB_PASSWORD": ("database", "password", str),
            "SITCHECK_DB_NAME": ("database", "database", str),
            "SITCHECK_DB_PORT": ("database", "port", int),
        }

        for env_key, (section, key, caster) in env_map.items():
            raw = os.getenv(env_key)
            if raw is None:
                continue
            config[section][key] = caster(raw)

    @staticmethod
    def _deep_update(target: Dict[str, Any], incoming: Dict[str, Any]):
        for key, value in incoming.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                ConfigManager._deep_update(target[key], value)
            else:
                target[key] = value

    @staticmethod
    def _parse_bool(raw: str) -> bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
