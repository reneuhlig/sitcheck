#!/usr/bin/env python3
"""
Startskript für YOLO26-Tracking basierte Live-Erkennung am Bibliothekseingang.
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict, Optional

from ConfigManager import ConfigManager


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Live people tracking + occupancy counting with Ultralytics YOLO track API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=os.path.join(SCRIPT_DIR, "config.yaml"),
        help="Pfad zur YAML-Konfiguration",
    )
    parser.add_argument("--video-source", default=None, help="Optionaler Override für Videoquelle")
    parser.add_argument("--show-window", action="store_true", help="Window explizit aktivieren")
    parser.add_argument("--headless", action="store_true", help="Window explizit deaktivieren")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    return parser.parse_args()


def _build_db_config(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    db = config["database"]
    if not db.get("enabled", False):
        return None
    return {
        "host": db["host"],
        "user": db["user"],
        "password": db["password"],
        "database": db["database"],
        "port": int(db["port"]),
    }

def _resolve_config_path(config_path: str) -> str:
    if os.path.isabs(config_path):
        return config_path
    return os.path.abspath(os.path.join(os.getcwd(), config_path))


def _resolve_relative_to_config(config_path: str, value: str) -> str:
    if not value:
        return value
    if os.path.isabs(value):
        return value
    config_dir = os.path.dirname(config_path)
    return os.path.abspath(os.path.join(config_dir, value))


def main():
    args = _parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Verzögert Importe von schweren/optionalen Abhängigkeiten, damit
    # `--help` ohne komplette Runtime-Dependencies funktioniert.
    from LiveProcessor import LiveProcessor
    from TrajectoryEntryAnalysisModule import EntranceZoneConfig
    from UltralyticsPersonDetector import UltralyticsPersonDetector

    resolved_config_path = _resolve_config_path(args.config)
    config_manager = ConfigManager(config_path=resolved_config_path)
    config = config_manager.load()

    if args.video_source:
        config["video"]["source"] = args.video_source
    if args.show_window:
        config["ui"]["show_window"] = True
    if args.headless:
        config["ui"]["show_window"] = False

    tracking_cfg = config.get("tracking", {})
    tracking_cfg["tracker"] = _resolve_relative_to_config(
        resolved_config_path,
        str(tracking_cfg.get("tracker", "bytetrack_entrance.yaml")),
    )
    config["tracking"] = tracking_cfg

    zone_config = EntranceZoneConfig.from_dict(config["zone"])
    db_config = _build_db_config(config)
    integration_config = dict(config.get("integration", {}) or {})

    def _persist_zone_update(updated_zone: EntranceZoneConfig):
        # Nicht-trivial: Änderungen aus der Live-UI sollen sofort wirksam und
        # reboot-sicher sein. Deshalb schreiben wir direkt zurück in die YAML.
        config_manager.update_zone(updated_zone.to_dict())

    try:
        detector = UltralyticsPersonDetector(
            api_url=str(config["tracking"].get("api_url", "")),
            api_key=str(config["tracking"].get("api_key", "")),
            confidence_threshold=float(config["tracking"]["confidence_threshold"]),
            device=str(config["tracking"].get("device", "cpu")),
            request_timeout_seconds=float(config["tracking"].get("api_timeout_seconds", 10.0)),
        )

        processor = LiveProcessor(
            detector=detector,
            video_source=str(config["video"]["source"]),
            fallback_source=config.get("video", {}).get("fallback_source"),
            hwaccel=str(config.get("video", {}).get("hwaccel", "auto")),
            youtube_cookies_from_browser=str(config.get("video", {}).get("youtube_cookies_from_browser", "")),
            youtube_cookiefile=_resolve_relative_to_config(
                resolved_config_path,
                str(config.get("video", {}).get("youtube_cookiefile", "")),
            ) if str(config.get("video", {}).get("youtube_cookiefile", "")).strip() else "",
            youtube_format=str(config.get("video", {}).get("youtube_format", "best[ext=mp4]/best")),
            youtube_player_client=str(config.get("video", {}).get("youtube_player_client", "android")),
            zone_config=zone_config,
            tracker_config=str(config["tracking"]["tracker"]),
            confidence_threshold=float(config["tracking"]["confidence_threshold"]),
            iou_threshold=float(config["tracking"]["iou_threshold"]),
            image_size=int(config["tracking"]["imgsz"]),
            tta_enabled=bool(config["tracking"].get("tta_enabled", False)),
            max_detections=int(config["tracking"].get("max_detections", 300)),
            stabilization_enabled=bool(config["tracking"].get("stabilization_enabled", True)),
            track_hold_frames=int(config["tracking"].get("track_hold_frames", 5)),
            box_ema_alpha=float(config["tracking"].get("box_ema_alpha", 0.65)),
            hold_confidence_decay=float(config["tracking"].get("hold_confidence_decay", 0.85)),
            trail_length=int(config["tracking"].get("trail_length", 12)),
            motion_min_pixels=float(config["tracking"].get("motion_min_pixels", 2.0)),
            process_every_n_frames=int(config["tracking"].get("process_every_n_frames", 1)),
            preprocess_enabled=bool(config.get("preprocess", {}).get("enabled", False)),
            preprocess_upscale=float(config.get("preprocess", {}).get("upscale", 1.0)),
            preprocess_clahe_clip=float(config.get("preprocess", {}).get("clahe_clip", 2.0)),
            preprocess_denoise=bool(config.get("preprocess", {}).get("denoise", False)),
            reconnect_delay=float(config["video"]["reconnect_delay"]),
            max_retries=int(config["video"]["max_retries"]),
            show_window=bool(config["ui"]["show_window"]),
            window_name=str(config["ui"].get("window_name", "Library Entry Tracking")),
            enable_zone_editor=bool(config["ui"].get("enable_zone_editor", True)),
            on_zone_changed=_persist_zone_update,
            db_config=db_config,
            integration_config=integration_config,
            config_dir=os.path.dirname(resolved_config_path),
        )
        processor.start()

    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        logger.error(f"Kritischer Fehler: {exc}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
