from __future__ import annotations

import copy
import os
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "video": {
        "source": "0",
        "fallback_source": "",
        "reconnect_delay": 1.0,
        "max_retries": 0,
        "hwaccel": "auto",
        "youtube_cookies_from_browser": "",
        "youtube_cookiefile": "",
        "youtube_format": "best[ext=mp4]/best",
        "youtube_player_client": "android",
        "input_mode": "youtube",
        "simulation": {
            "directory": "LiveFeed Simulation",
            "control_mode": "remote_control",
            "default_clip_id": "leerlauf",
            "idle_loop": True,
        },
    },
    "tracking": {
        "detector_mode": "hybrid",
        "api_url": "",
        "api_key": "",
        "api_timeout_seconds": 10.0,
        "api_failure_cooldown_seconds": 2.0,
        "max_api_fps": 10.0,
        "api_jpeg_quality": 85,
        "model_path": "models/yolo28n.pt",
        "hybrid_target_fps": 20.0,
        "api_refresh_every_n_frames": 2,
        "local_preload_enabled": True,
        "local_fill_enabled": True,
        "local_fill_max_fps": 10.0,
        "local_refresh_every_n_frames": 2,
        "cache_fallback_max_age_frames": 4,
        "min_person_height_ratio": 0.10,
        "min_person_area_ratio": 0.010,
        "min_person_aspect_ratio": 1.00,
        "allow_classless_person_detections": False,
        "device": "cpu",
        "tracker": "bytetrack_entrance.yaml",
        "confidence_threshold": 0.1,
        "iou_threshold": 0.45,
        "imgsz": 640,
        "analysis_roi": {
            "enabled": False,
            "mode": "rect",
            "x_min": 0.0,
            "y_min": 0.0,
            "x_max": 1.0,
            "y_max": 1.0,
            "polygon_points": [],
        },
        "tta_enabled": False,
        "max_detections": 300,
        "stabilization_enabled": True,
        "track_hold_frames": 5,
        "box_ema_alpha": 0.65,
        "hold_confidence_decay": 0.85,
        "trail_length": 12,
        "motion_min_pixels": 2.0,
        "process_every_n_frames": 1,
    },
    "preprocess": {
        "enabled": False,
        "upscale": 1.0,
        "clahe_clip": 2.0,
        "denoise": False,
    },
    "dashboard": {
        "stream_fps": 20,
        "capture_max_fps": 20,
        "visual_update_fps": 20,
        "jpeg_quality": 40,
        "jpeg_optimize": False,
        "stream_max_width": 640,
        "analysis_queue_frames": 64,
        "analysis_skip_threshold_frames": 48,
        "dynamic_skip_enabled": True,
        "dynamic_skip_queue_threshold": 40,
        "dynamic_skip_max_n": 1,
        "profile_simulation": {
            "enabled": False,
            "excel_path": "KI_Projekt_Daten_einJahr.xlsx",
            "tick_seconds": 60.0,
            "profile_blend": 0.72,
            "noise_sigma_scale": 0.85,
            "max_step_per_tick": 2.0,
            "rollback_minutes": 15.0,
        },
        "dash": {
            "enabled": False,
            "output_dir": "runtime/dash",
            "segment_time": 1.0,
            "list_size": 12,
            "preset": "ultrafast",
            "tune": "zerolatency",
            "crf": 36,
            "hwaccel": "auto",
            "vaapi_device": "/dev/dri/renderD128",
            "abr": {
                "enabled": True,
                "high_bitrate_kbps": 1400,
                "low_bitrate_kbps": 650,
                "low_scale": 0.6,
            },
        },
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
    "integration": {
        "prognose_db": {
            "enabled": False,
            "api_ingest_enabled": False,
            "api_base_url": "http://127.0.0.1:8000",
            "api_timeout_seconds": 3.0,
            "host": "127.0.0.1",
            "user": "sitcheck",
            "password": "change_me",
            "database": "sitcheck",
            "port": 5432,
            "zone_id": "default-zone",
            "source": "vision-direct-db",
            "write_mode": "frame_near",
            "max_writes_per_second": 2,
            "strict_zone_check": True,
            "ensure_zone": False,
            "default_zone_capacity": 100,
            "capacity_refresh_seconds": 30,
            "flush_batch_size": 200,
            "log_path": "../website-dashboard/runtime/logs/integration_writer.log",
            "spool": {
                "enabled": True,
                "path": "../website-dashboard/runtime/logs/prognose_counts_spool.jsonl",
                "max_entries": 20000,
            },
        }
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
            "SITCHECK_VIDEO_FALLBACK_SOURCE": ("video", "fallback_source", str),
            "SITCHECK_SIM_IDLE_LOOP": ("video", "simulation.idle_loop", self._parse_bool),
            "SITCHECK_DETECTOR_MODE": ("tracking", "detector_mode", str),
            "SITCHECK_PREDICT_URL": ("tracking", "api_url", str),
            "SITCHECK_PREDICT_API_KEY": ("tracking", "api_key", str),
            "SITCHECK_PREDICT_TIMEOUT_SECONDS": ("tracking", "api_timeout_seconds", float),
            "SITCHECK_YOLO_MODEL": ("tracking", "model_path", str),
            "SITCHECK_HYBRID_TARGET_FPS": ("tracking", "hybrid_target_fps", float),
            "SITCHECK_API_REFRESH_EVERY_N_FRAMES": ("tracking", "api_refresh_every_n_frames", int),
            "SITCHECK_LOCAL_PRELOAD_ENABLED": ("tracking", "local_preload_enabled", self._parse_bool),
            "SITCHECK_LOCAL_FILL_ENABLED": ("tracking", "local_fill_enabled", self._parse_bool),
            "SITCHECK_LOCAL_FILL_MAX_FPS": ("tracking", "local_fill_max_fps", float),
            "SITCHECK_LOCAL_REFRESH_EVERY_N_FRAMES": ("tracking", "local_refresh_every_n_frames", int),
            "SITCHECK_CACHE_FALLBACK_MAX_AGE_FRAMES": ("tracking", "cache_fallback_max_age_frames", int),
            "SITCHECK_MIN_PERSON_HEIGHT_RATIO": ("tracking", "min_person_height_ratio", float),
            "SITCHECK_MIN_PERSON_AREA_RATIO": ("tracking", "min_person_area_ratio", float),
            "SITCHECK_MIN_PERSON_ASPECT_RATIO": ("tracking", "min_person_aspect_ratio", float),
            "SITCHECK_ALLOW_CLASSLESS_PERSON_DETECTIONS": ("tracking", "allow_classless_person_detections", self._parse_bool),
            "SITCHECK_DEVICE": ("tracking", "device", str),
            "SITCHECK_TRACKER": ("tracking", "tracker", str),
            "SITCHECK_CONFIDENCE": ("tracking", "confidence_threshold", float),
            "SITCHECK_IOU": ("tracking", "iou_threshold", float),
            "SITCHECK_IMGSZ": ("tracking", "imgsz", int),
            "SITCHECK_ANALYSIS_ROI_ENABLED": ("tracking", "analysis_roi.enabled", self._parse_bool),
            "SITCHECK_ANALYSIS_ROI_X_MIN": ("tracking", "analysis_roi.x_min", float),
            "SITCHECK_ANALYSIS_ROI_Y_MIN": ("tracking", "analysis_roi.y_min", float),
            "SITCHECK_ANALYSIS_ROI_X_MAX": ("tracking", "analysis_roi.x_max", float),
            "SITCHECK_ANALYSIS_ROI_Y_MAX": ("tracking", "analysis_roi.y_max", float),
            "SITCHECK_TTA_ENABLED": ("tracking", "tta_enabled", self._parse_bool),
            "SITCHECK_MAX_DETECTIONS": ("tracking", "max_detections", int),
            "SITCHECK_STABILIZATION_ENABLED": ("tracking", "stabilization_enabled", self._parse_bool),
            "SITCHECK_TRACK_HOLD_FRAMES": ("tracking", "track_hold_frames", int),
            "SITCHECK_BOX_EMA_ALPHA": ("tracking", "box_ema_alpha", float),
            "SITCHECK_HOLD_CONFIDENCE_DECAY": ("tracking", "hold_confidence_decay", float),
            "SITCHECK_TRAIL_LENGTH": ("tracking", "trail_length", int),
            "SITCHECK_MOTION_MIN_PIXELS": ("tracking", "motion_min_pixels", float),
            "SITCHECK_PROCESS_EVERY_N_FRAMES": ("tracking", "process_every_n_frames", int),
            "SITCHECK_PREPROCESS_ENABLED": ("preprocess", "enabled", self._parse_bool),
            "SITCHECK_PREPROCESS_UPSCALE": ("preprocess", "upscale", float),
            "SITCHECK_PREPROCESS_CLAHE_CLIP": ("preprocess", "clahe_clip", float),
            "SITCHECK_PREPROCESS_DENOISE": ("preprocess", "denoise", self._parse_bool),
            "SITCHECK_DASH_STREAM_FPS": ("dashboard", "stream_fps", int),
            "SITCHECK_PROFILE_SIM_ENABLED": ("dashboard", "profile_simulation.enabled", self._parse_bool),
            "SITCHECK_PROFILE_SIM_EXCEL_PATH": ("dashboard", "profile_simulation.excel_path", str),
            "SITCHECK_PROFILE_SIM_TICK_SECONDS": ("dashboard", "profile_simulation.tick_seconds", float),
            "SITCHECK_PROFILE_SIM_BLEND": ("dashboard", "profile_simulation.profile_blend", float),
            "SITCHECK_PROFILE_SIM_NOISE_SCALE": ("dashboard", "profile_simulation.noise_sigma_scale", float),
            "SITCHECK_PROFILE_SIM_MAX_STEP": ("dashboard", "profile_simulation.max_step_per_tick", float),
            "SITCHECK_PROFILE_SIM_ROLLBACK_MINUTES": ("dashboard", "profile_simulation.rollback_minutes", float),
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
            "SITCHECK_PROGNOSE_DB_ENABLED": ("integration", "prognose_db.enabled", self._parse_bool),
            "SITCHECK_PROGNOSE_API_INGEST_ENABLED": ("integration", "prognose_db.api_ingest_enabled", self._parse_bool),
            "SITCHECK_PROGNOSE_API_BASE_URL": ("integration", "prognose_db.api_base_url", str),
            "SITCHECK_PROGNOSE_API_TIMEOUT_SECONDS": ("integration", "prognose_db.api_timeout_seconds", float),
            "SITCHECK_PROGNOSE_DB_HOST": ("integration", "prognose_db.host", str),
            "SITCHECK_PROGNOSE_DB_USER": ("integration", "prognose_db.user", str),
            "SITCHECK_PROGNOSE_DB_PASSWORD": ("integration", "prognose_db.password", str),
            "SITCHECK_PROGNOSE_DB_NAME": ("integration", "prognose_db.database", str),
            "SITCHECK_PROGNOSE_DB_PORT": ("integration", "prognose_db.port", int),
            "SITCHECK_PROGNOSE_DB_ZONE_ID": ("integration", "prognose_db.zone_id", str),
            "SITCHECK_PROGNOSE_DB_SOURCE": ("integration", "prognose_db.source", str),
            "SITCHECK_PROGNOSE_DB_WRITE_MODE": ("integration", "prognose_db.write_mode", str),
            "SITCHECK_PROGNOSE_DB_MAX_WPS": ("integration", "prognose_db.max_writes_per_second", float),
            "SITCHECK_PROGNOSE_DB_STRICT_ZONE_CHECK": ("integration", "prognose_db.strict_zone_check", self._parse_bool),
            "SITCHECK_PROGNOSE_DB_ENSURE_ZONE": ("integration", "prognose_db.ensure_zone", self._parse_bool),
            "SITCHECK_PROGNOSE_DB_DEFAULT_CAPACITY": ("integration", "prognose_db.default_zone_capacity", int),
            "SITCHECK_PROGNOSE_DB_CAPACITY_REFRESH_SECONDS": ("integration", "prognose_db.capacity_refresh_seconds", int),
            "SITCHECK_PROGNOSE_DB_FLUSH_BATCH_SIZE": ("integration", "prognose_db.flush_batch_size", int),
            "SITCHECK_PROGNOSE_DB_LOG_PATH": ("integration", "prognose_db.log_path", str),
            "SITCHECK_PROGNOSE_DB_SPOOL_ENABLED": ("integration", "prognose_db.spool.enabled", self._parse_bool),
            "SITCHECK_PROGNOSE_DB_SPOOL_PATH": ("integration", "prognose_db.spool.path", str),
            "SITCHECK_PROGNOSE_DB_SPOOL_MAX_ENTRIES": ("integration", "prognose_db.spool.max_entries", int),
        }

        for env_key, (section, key, caster) in env_map.items():
            raw = os.getenv(env_key)
            if raw is None:
                continue
            if "." in key:
                parts = key.split(".")
                target = config[section]
                for part in parts[:-1]:
                    if part not in target or not isinstance(target.get(part), dict):
                        target[part] = {}
                    target = target[part]
                target[parts[-1]] = caster(raw)
            else:
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
