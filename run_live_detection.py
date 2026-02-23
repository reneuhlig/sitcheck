#!/usr/bin/env python3
"""
Startskript für YOLO26-Tracking basierte Live-Erkennung am Bibliothekseingang.
"""

import argparse
import logging
import sys
from typing import Any, Dict, Optional

from ConfigManager import ConfigManager
from LiveProcessor import LiveProcessor
from TrajectoryEntryAnalysisModule import EntranceZoneConfig
from UltralyticsPersonDetector import UltralyticsPersonDetector


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Live people tracking + occupancy counting with Ultralytics YOLO track API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml", help="Pfad zur YAML-Konfiguration")
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


def main():
    args = _parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config_manager = ConfigManager(config_path=args.config)
    config = config_manager.load()

    if args.video_source:
        config["video"]["source"] = args.video_source
    if args.show_window:
        config["ui"]["show_window"] = True
    if args.headless:
        config["ui"]["show_window"] = False

    zone_config = EntranceZoneConfig.from_dict(config["zone"])
    db_config = _build_db_config(config)

    def _persist_zone_update(updated_zone: EntranceZoneConfig):
        # Nicht-trivial: Änderungen aus der Live-UI sollen sofort wirksam und
        # reboot-sicher sein. Deshalb schreiben wir direkt zurück in die YAML.
        config_manager.update_zone(updated_zone.to_dict())

    try:
        detector = UltralyticsPersonDetector(
            model_path=config["tracking"]["model_path"],
            confidence_threshold=float(config["tracking"]["confidence_threshold"]),
            device=str(config["tracking"].get("device", "cpu")),
        )

        processor = LiveProcessor(
            detector=detector,
            video_source=str(config["video"]["source"]),
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
