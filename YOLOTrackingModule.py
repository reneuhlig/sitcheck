from typing import Any, Dict, List

import cv2


class YOLOTrackingModule:
    """Kapselt die YOLO-Tracking-Inferenz (Ultralytics model.track)."""

    def __init__(
        self,
        detector,
        tracker_config: str = "bytetrack.yaml",
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        image_size: int = 640,
        tta_enabled: bool = False,
        max_detections: int = 300,
        stabilization_enabled: bool = True,
        track_hold_frames: int = 5,
        box_ema_alpha: float = 0.65,
        hold_confidence_decay: float = 0.85,
        trail_length: int = 12,
        motion_min_pixels: float = 2.0,
        preprocess_enabled: bool = False,
        preprocess_upscale: float = 1.0,
        preprocess_clahe_clip: float = 2.0,
        preprocess_denoise: bool = False,
    ):
        self.detector = detector
        self.tracker_config = tracker_config
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.tta_enabled = tta_enabled
        self.max_detections = max(1, int(max_detections))
        self.stabilization_enabled = stabilization_enabled
        self.track_hold_frames = max(0, int(track_hold_frames))
        self.box_ema_alpha = max(0.05, min(1.0, float(box_ema_alpha)))
        self.hold_confidence_decay = max(0.1, min(1.0, float(hold_confidence_decay)))
        self.trail_length = max(2, int(trail_length))
        self.motion_min_pixels = max(0.0, float(motion_min_pixels))
        self.preprocess_enabled = preprocess_enabled
        self.preprocess_upscale = preprocess_upscale
        self.preprocess_clahe_clip = preprocess_clahe_clip
        self.preprocess_denoise = preprocess_denoise
        self._clahe = cv2.createCLAHE(
            clipLimit=max(1.0, float(self.preprocess_clahe_clip)),
            tileGridSize=(8, 8),
        )
        self._frame_index = 0
        self._track_memory: Dict[int, Dict[str, Any]] = {}

    def _preprocess_frame(self, frame):
        processed = frame

        if self.preprocess_upscale > 1.01:
            processed = cv2.resize(
                processed,
                None,
                fx=self.preprocess_upscale,
                fy=self.preprocess_upscale,
                interpolation=cv2.INTER_CUBIC,
            )

        ycrcb = cv2.cvtColor(processed, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y = self._clahe.apply(y)
        ycrcb = cv2.merge([y, cr, cb])
        processed = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

        if self.preprocess_denoise:
            processed = cv2.fastNlMeansDenoisingColored(processed, None, 5, 5, 7, 21)

        return processed

    def track(self, frame) -> List[Dict[str, Any]]:
        self._frame_index += 1
        model_input = self._preprocess_frame(frame) if self.preprocess_enabled else frame
        raw_tracks = self.detector.track(
            frame=model_input,
            tracker=self.tracker_config,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            augment=self.tta_enabled,
            max_det=self.max_detections,
        )

        if not self.stabilization_enabled:
            return raw_tracks

        return self._stabilize_tracks(raw_tracks)

    def _stabilize_tracks(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        visible_track_ids = set()
        stable_tracks: List[Dict[str, Any]] = []

        for track in tracks:
            track_id = track.get("track_id")
            bbox = track.get("bbox")
            center = track.get("center")
            confidence = float(track.get("confidence", 0.0))

            if track_id is None or bbox is None or center is None:
                continue

            visible_track_ids.add(track_id)
            prev = self._track_memory.get(track_id)

            if prev:
                bbox = self._blend_bbox(prev["bbox"], bbox, self.box_ema_alpha)
                center = self._blend_center(prev["center"], center, self.box_ema_alpha)

            if prev:
                motion_vector = (
                    float(center[0]) - float(prev["center"][0]),
                    float(center[1]) - float(prev["center"][1]),
                )
            else:
                motion_vector = (0.0, 0.0)

            motion_magnitude = ((motion_vector[0] ** 2) + (motion_vector[1] ** 2)) ** 0.5
            motion_direction = self._direction_label(motion_vector, motion_magnitude)

            trail = list(prev.get("trail", [])) if prev else []
            trail.append((float(center[0]), float(center[1])))
            if len(trail) > self.trail_length:
                trail = trail[-self.trail_length :]

            self._track_memory[track_id] = {
                "bbox": bbox,
                "center": center,
                "confidence": confidence,
                "motion_vector": motion_vector,
                "motion_magnitude": motion_magnitude,
                "motion_direction": motion_direction,
                "trail": trail,
                "last_seen": self._frame_index,
            }

            stable_tracks.append(
                {
                    **track,
                    "bbox": bbox,
                    "center": center,
                    "confidence": confidence,
                    "motion_vector": motion_vector,
                    "motion_magnitude": motion_magnitude,
                    "motion_direction": motion_direction,
                    "trail": trail,
                    "is_stale": False,
                }
            )

        min_hold_confidence = max(0.03, self.confidence_threshold * 0.4)
        for track_id, state in list(self._track_memory.items()):
            if track_id in visible_track_ids:
                continue

            missed_frames = self._frame_index - int(state["last_seen"])
            if missed_frames > self.track_hold_frames:
                del self._track_memory[track_id]
                continue

            decayed_conf = float(state["confidence"]) * (self.hold_confidence_decay ** missed_frames)
            if decayed_conf < min_hold_confidence:
                del self._track_memory[track_id]
                continue

            stable_tracks.append(
                {
                    "track_id": track_id,
                    "bbox": state["bbox"],
                    "center": state["center"],
                    "confidence": decayed_conf,
                    "motion_vector": state.get("motion_vector", (0.0, 0.0)),
                    "motion_magnitude": float(state.get("motion_magnitude", 0.0)),
                    "motion_direction": state.get("motion_direction", "still"),
                    "trail": list(state.get("trail", [])),
                    "is_stale": True,
                }
            )

        stable_tracks.sort(key=lambda item: int(item.get("track_id", -1)))
        return stable_tracks

    @staticmethod
    def _blend_bbox(previous_bbox, current_bbox, alpha: float):
        if len(previous_bbox) != 4 or len(current_bbox) != 4:
            return current_bbox
        return [
            (alpha * float(current_bbox[idx])) + ((1.0 - alpha) * float(previous_bbox[idx]))
            for idx in range(4)
        ]

    @staticmethod
    def _blend_center(previous_center, current_center, alpha: float):
        if len(previous_center) != 2 or len(current_center) != 2:
            return current_center
        return (
            (alpha * float(current_center[0])) + ((1.0 - alpha) * float(previous_center[0])),
            (alpha * float(current_center[1])) + ((1.0 - alpha) * float(previous_center[1])),
        )

    def _direction_label(self, motion_vector, motion_magnitude: float) -> str:
        if motion_magnitude < self.motion_min_pixels:
            return "still"

        dx, dy = float(motion_vector[0]), float(motion_vector[1])
        abs_dx = abs(dx)
        abs_dy = abs(dy)

        if abs_dx > abs_dy * 1.35:
            return "right" if dx > 0 else "left"
        if abs_dy > abs_dx * 1.35:
            return "down" if dy > 0 else "up"

        if dx >= 0 and dy >= 0:
            return "down-right"
        if dx >= 0 and dy < 0:
            return "up-right"
        if dx < 0 and dy >= 0:
            return "down-left"
        return "up-left"
