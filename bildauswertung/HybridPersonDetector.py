import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from BaseDetector import BaseDetector


class HybridPersonDetector(BaseDetector):
    """Local-first detector: lokales YOLO primär, API nur als Fallback bei Local-Failure."""

    def __init__(
        self,
        api_detector: Optional[BaseDetector] = None,
        local_detector: Optional[BaseDetector] = None,
        local_fill_enabled: bool = True,
        local_fill_max_fps: float = 4.0,
        cache_fallback_max_age_frames: int = 30,
        api_refresh_every_n_frames: int = 4,
        local_refresh_every_n_frames: int = 8,
        api_result_max_age_frames: int = 6,
        max_cache_only_frames: int = 2,
    ):
        if api_detector is None and local_detector is None:
            raise ValueError("HybridPersonDetector braucht mindestens einen Detector")
        super().__init__("Hybrid API+Local YOLO", "v1")
        self.api_detector = api_detector
        self.local_detector = local_detector
        self.local_fill_enabled = bool(local_fill_enabled and local_detector is not None)
        self.local_fill_max_fps = max(0.0, float(local_fill_max_fps))
        self.cache_fallback_max_age_frames = max(0, int(cache_fallback_max_age_frames))
        self.api_refresh_every_n_frames = max(1, int(api_refresh_every_n_frames))
        self.local_refresh_every_n_frames = max(1, int(local_refresh_every_n_frames))
        self.api_result_max_age_frames = max(0, int(api_result_max_age_frames))
        self.max_cache_only_frames = max(0, int(max_cache_only_frames))
        self.last_track_ok = True
        self.last_track_error = ""
        self.last_detector_source = "init"
        self._last_local_refresh_ts = 0.0
        self._last_api_refresh_frame = -1_000_000
        self._last_local_refresh_frame = -1_000_000
        self._api_future_frame_counter = -1_000_000
        self._api_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hybrid-api-detector")
        self._api_future: Optional[Future] = None
        self._cached_tracks: List[Dict[str, Any]] = []
        self._cached_frame_counter = 0
        self._frame_counter = 0
        self._cache_only_streak = 0
        self._local_fail_streak = 0
        self._local_fail_logged = False

    def _api_ready(self, now_ts: float) -> bool:
        if self.api_detector is None:
            return False
        if (self._frame_counter - self._last_api_refresh_frame) < self.api_refresh_every_n_frames:
            return False
        cooldown_until = float(getattr(self.api_detector, "_failure_cooldown_until_ts", 0.0) or 0.0)
        if cooldown_until > now_ts:
            return False
        min_interval = float(getattr(self.api_detector, "_min_api_interval_seconds", 0.0) or 0.0)
        last_request = float(getattr(self.api_detector, "_last_api_request_ts", 0.0) or 0.0)
        return min_interval <= 0.0 or (now_ts - last_request) >= min_interval

    def _local_budget_ready(self, now_ts: float) -> bool:
        if self.local_fill_max_fps <= 0.0:
            return True
        min_interval = 1.0 / self.local_fill_max_fps
        return (now_ts - self._last_local_refresh_ts) >= min_interval

    def _local_ready(self, now_ts: float, ignore_frame_interval: bool = False) -> bool:
        if not self.local_fill_enabled or self.local_detector is None:
            return False
        if (not ignore_frame_interval) and (
            (self._frame_counter - self._last_local_refresh_frame) < self.local_refresh_every_n_frames
        ):
            return False
        return self._local_budget_ready(now_ts)

    def _cache_tracks(self, tracks: List[Dict[str, Any]], source: str):
        self._cached_tracks = [dict(track) for track in tracks]
        self._cached_frame_counter = self._frame_counter
        self.last_detector_source = source

    def _cached_tracks_still_valid(self) -> bool:
        if not self._cached_tracks:
            return False
        if self.cache_fallback_max_age_frames <= 0:
            return True
        return (self._frame_counter - self._cached_frame_counter) <= self.cache_fallback_max_age_frames

    def _return_cached_tracks(self) -> List[Dict[str, Any]]:
        self.last_track_ok = True
        self.last_track_error = ""
        self.last_detector_source = "cache"
        self._cache_only_streak += 1
        return [dict(track) for track in self._cached_tracks]

    def _run_api_track(
        self,
        frame,
        tracker: str,
        conf: float,
        iou: float,
        imgsz: int,
        augment: bool,
        max_det: int,
    ) -> Dict[str, Any]:
        tracks = self.api_detector.track(
            frame=frame,
            tracker=tracker,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            augment=augment,
            max_det=max_det,
        )
        return {
            "tracks": tracks,
            "ok": bool(getattr(self.api_detector, "last_track_ok", True)),
            "error": str(getattr(self.api_detector, "last_track_error", "") or ""),
        }

    def _consume_api_future(self) -> Optional[List[Dict[str, Any]]]:
        if self._api_future is None or not self._api_future.done():
            return None
        future = self._api_future
        self._api_future = None
        try:
            result = future.result()
        except Exception as exc:
            self.last_track_ok = False
            self.last_track_error = str(exc)
            self.last_detector_source = "api_error"
            return None

        if (
            self.api_result_max_age_frames > 0
            and (self._frame_counter - self._api_future_frame_counter) > self.api_result_max_age_frames
        ):
            self.last_track_ok = False
            self.last_track_error = "api_result_stale"
            self.last_detector_source = "api_stale"
            return None

        tracks = result.get("tracks", [])
        if bool(result.get("ok", False)):
            self.last_track_ok = True
            self.last_track_error = ""
            self._cache_tracks(tracks, "api")
            self._cache_only_streak = 0
            return [dict(track) for track in tracks]

        self.last_track_ok = False
        self.last_track_error = str(result.get("error", "") or "api_failed")
        self.last_detector_source = "api_error"
        return None

    def detect(self, image) -> Dict[str, Any]:
        tracks = self.track(frame=image)
        confidences = [float(item.get("confidence", 0.0)) for item in tracks]
        return {
            "persons_detected": len(tracks),
            "persons": tracks,
            "confidences": confidences,
            "avg_confidence": (sum(confidences) / len(confidences)) if confidences else 0.0,
            "max_confidence": max(confidences) if confidences else 0.0,
            "min_confidence": min(confidences) if confidences else 0.0,
        }

    def track(
        self,
        frame,
        tracker: str = "bytetrack.yaml",
        conf: float = 0.4,
        iou: float = 0.5,
        imgsz: int = 640,
        augment: bool = False,
        max_det: int = 300,
    ) -> List[Dict[str, Any]]:
        self._frame_counter += 1
        now_ts = time.time()

        # ── 1. LOCAL ZUERST (primäre Quelle) ──────────────────────────
        if self._local_ready(now_ts, ignore_frame_interval=True):
            self._last_local_refresh_ts = now_ts
            self._last_local_refresh_frame = self._frame_counter
            tracks = self.local_detector.track(
                frame=frame,
                tracker=tracker,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                augment=False,
                max_det=max_det,
            )
            if bool(getattr(self.local_detector, "last_track_ok", True)):
                self.last_track_ok = True
                self.last_track_error = ""
                self._cache_tracks(tracks, "local")
                self._cache_only_streak = 0
                self._local_fail_streak = 0
                return tracks

            self._local_fail_streak += 1
            if self._local_fail_streak >= 10 and not self._local_fail_logged:
                self._local_fail_logged = True
                print(
                    f"[WARN] Lokaler Detektor {self._local_fail_streak}x fehlgeschlagen: "
                    f"{getattr(self.local_detector, 'last_track_error', '?')}"
                )

        # ── 2. API als Fallback (nur wenn Local versagt) ──────────────
        api_tracks = self._consume_api_future()
        if api_tracks is not None:
            self._cache_tracks(api_tracks, "api_fallback")
            self._cache_only_streak = 0
            return api_tracks

        if self._local_fail_streak >= 3 and self._api_ready(now_ts) and self._api_future is None:
            self._last_api_refresh_frame = self._frame_counter
            self._api_future_frame_counter = self._frame_counter
            self._api_future = self._api_executor.submit(
                self._run_api_track,
                frame.copy(),
                tracker,
                conf,
                iou,
                imgsz,
                augment,
                max_det,
            )

        # ── 3. Cache als letzter Fallback ─────────────────────────────
        if self._cached_tracks_still_valid():
            return self._return_cached_tracks()

        self.last_track_ok = self._local_fail_streak == 0
        self.last_track_error = "local_and_api_unavailable" if self._local_fail_streak > 0 else ""
        self.last_detector_source = "empty"
        self._cache_only_streak += 1
        return []

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "framework": "Hybrid API/local/cached YOLO",
            "api": self.api_detector.get_model_info() if self.api_detector else None,
            "local": self.local_detector.get_model_info() if self.local_detector else None,
            "local_fill_enabled": self.local_fill_enabled,
            "local_fill_max_fps": self.local_fill_max_fps,
            "api_refresh_every_n_frames": self.api_refresh_every_n_frames,
            "local_refresh_every_n_frames": self.local_refresh_every_n_frames,
            "api_result_max_age_frames": self.api_result_max_age_frames,
            "max_cache_only_frames": self.max_cache_only_frames,
        }
