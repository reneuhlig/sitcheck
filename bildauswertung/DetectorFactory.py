import os
import math
from typing import Any, Dict, Optional

from HybridPersonDetector import HybridPersonDetector
from LocalYoloPersonDetector import LocalYoloPersonDetector
from UltralyticsPersonDetector import UltralyticsPersonDetector


def _resolve_path(config_dir: Optional[str], path_value: str) -> str:
    value = str(path_value or "").strip()
    if not value:
        return value
    if os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(config_dir or ".", value))


def build_person_detector(tracking_cfg: Dict[str, Any], config_dir: Optional[str] = None):
    mode = str(tracking_cfg.get("detector_mode", "hybrid")).strip().lower()
    if mode not in {"api", "local", "hybrid"}:
        mode = "hybrid"

    common = {
        "confidence_threshold": float(tracking_cfg.get("confidence_threshold", 0.5)),
        "device": str(tracking_cfg.get("device", "cpu")),
        "min_person_height_ratio": float(tracking_cfg.get("min_person_height_ratio", 0.10)),
        "min_person_area_ratio": float(tracking_cfg.get("min_person_area_ratio", 0.010)),
        "min_person_aspect_ratio": float(tracking_cfg.get("min_person_aspect_ratio", 1.00)),
    }

    api_detector = None
    if mode in {"api", "hybrid"}:
        api_url = str(tracking_cfg.get("api_url", "")).strip()
        api_key = str(tracking_cfg.get("api_key", "")).strip()
        if api_url and api_key:
            api_detector = UltralyticsPersonDetector(
                api_url=api_url,
                api_key=api_key,
                request_timeout_seconds=float(tracking_cfg.get("api_timeout_seconds", 10.0)),
                failure_cooldown_seconds=float(tracking_cfg.get("api_failure_cooldown_seconds", 2.0)),
                max_api_fps=float(tracking_cfg.get("max_api_fps", 0.0)),
                jpeg_quality=int(tracking_cfg.get("api_jpeg_quality", 85)),
                allow_classless_person_detections=bool(tracking_cfg.get("allow_classless_person_detections", False)),
                **common,
            )
        elif mode == "api":
            raise ValueError("tracking.detector_mode=api braucht api_url und api_key")

    local_detector = None
    if mode in {"local", "hybrid"}:
        model_path = _resolve_path(config_dir, str(tracking_cfg.get("model_path", "models/yolo26n.pt")))
        if model_path and os.path.exists(model_path):
            local_detector = LocalYoloPersonDetector(
                model_path=model_path,
                **common,
            )
            if bool(tracking_cfg.get("local_preload_enabled", True)):
                try:
                    local_detector.warmup(imgsz=int(tracking_cfg.get("imgsz", 640)))
                    print(f"[INFO] Lokales YOLO-Modell geladen: {model_path}")
                except Exception as exc:
                    print(f"[WARN] Lokales YOLO-Modell warmup fehlgeschlagen: {exc}")
                    local_detector = None
        else:
            print(f"[WARN] Lokales Modell nicht gefunden: {model_path}")
        if mode == "local" and local_detector is None:
            raise ValueError(f"tracking.detector_mode=local braucht ein lokales Modell: {model_path}")

    if mode == "api":
        return api_detector
    if mode == "local":
        return local_detector
    if api_detector and local_detector:
        target_fps = max(1.0, float(tracking_cfg.get("hybrid_target_fps", 30.0)))
        api_max_fps = max(0.0, float(tracking_cfg.get("max_api_fps", 0.0)))
        local_max_fps = max(0.0, float(tracking_cfg.get("local_fill_max_fps", 4.0)))
        default_api_refresh_every_n = max(1, math.ceil(target_fps / api_max_fps)) if api_max_fps > 0.0 else 1
        default_local_refresh_every_n = max(1, math.ceil(target_fps / local_max_fps)) if local_max_fps > 0.0 else 1
        configured_api_refresh_every_n = int(tracking_cfg.get("api_refresh_every_n_frames", default_api_refresh_every_n))
        configured_local_refresh_every_n = int(
            tracking_cfg.get("local_refresh_every_n_frames", default_local_refresh_every_n)
        )
        # Never allow configured refresh cadence to be sparser than the effective API/local FPS budgets.
        api_refresh_every_n_frames = max(1, min(configured_api_refresh_every_n, default_api_refresh_every_n))
        local_refresh_every_n_frames = max(1, min(configured_local_refresh_every_n, default_local_refresh_every_n))
        print(
            f"[INFO] HybridDetector: API jeder {api_refresh_every_n_frames}. Frame, "
            f"Local jeder {local_refresh_every_n_frames}. Frame, target {target_fps}fps"
        )
        return HybridPersonDetector(
            api_detector=api_detector,
            local_detector=local_detector,
            local_fill_enabled=bool(tracking_cfg.get("local_fill_enabled", True)),
            local_fill_max_fps=float(tracking_cfg.get("local_fill_max_fps", 4.0)),
            cache_fallback_max_age_frames=int(tracking_cfg.get("cache_fallback_max_age_frames", 30)),
            api_refresh_every_n_frames=api_refresh_every_n_frames,
            local_refresh_every_n_frames=local_refresh_every_n_frames,
            api_result_max_age_frames=int(tracking_cfg.get("api_result_max_age_frames", 6)),
            max_cache_only_frames=int(tracking_cfg.get("max_cache_only_frames", 2)),
        )
    detector = api_detector or local_detector
    if detector is None:
        raise ValueError("Kein Detector verfügbar: API-Konfiguration fehlt und lokales Modell wurde nicht gefunden")
    if api_detector and not local_detector:
        print("[WARN] Nur API-Detektor aktiv – kein lokales Modell verfügbar!")
    elif local_detector and not api_detector:
        print("[WARN] Nur lokaler Detektor aktiv – keine API konfiguriert!")
    return detector
