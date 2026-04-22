import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from BaseDetector import BaseDetector


class LocalYoloPersonDetector(BaseDetector):
    """Lightweight local YOLO detector with simple persistent ID assignment."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        device: str = "cpu",
        min_person_height_ratio: float = 0.10,
        min_person_area_ratio: float = 0.010,
        min_person_aspect_ratio: float = 1.00,
    ):
        normalized_path = os.path.abspath(str(model_path or "").strip())
        if not normalized_path or not os.path.exists(normalized_path):
            raise ValueError(f"tracking.model_path nicht gefunden: {normalized_path}")

        super().__init__("Local YOLO", os.path.basename(normalized_path))
        self.model_path = normalized_path
        self.confidence_threshold = float(confidence_threshold)
        normalized_device = str(device).strip().lower()
        self.device = "cpu" if normalized_device in {"", "auto"} else normalized_device
        self.min_person_height_ratio = max(0.0, min(1.0, float(min_person_height_ratio)))
        self.min_person_area_ratio = max(0.0, min(1.0, float(min_person_area_ratio)))
        self.min_person_aspect_ratio = max(0.0, float(min_person_aspect_ratio))
        self.last_track_ok = True
        self.last_track_error = ""
        self.last_inference_seconds = 0.0
        self._model = None
        self._active_tracks: Dict[int, Dict[str, Any]] = {}
        self._next_track_id = 1
        self._max_lost_frames = 30
        self._frame_counter = 0

    @staticmethod
    def _xyxy_to_center(bbox: List[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _bbox_iou(a: List[float], b: List[float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter_area
        if denom <= 0.0:
            return 0.0
        return inter_area / denom

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        from ultralytics import YOLO

        self._model = YOLO(self.model_path)
        return self._model

    def warmup(self, imgsz: int = 640):
        model = self._ensure_model()
        size = max(64, int(imgsz))
        dummy = np.zeros((size, size, 3), dtype=np.uint8)
        model.predict(
            dummy,
            conf=max(0.01, self.confidence_threshold),
            imgsz=size,
            classes=[0],
            device=self.device,
            verbose=False,
        )

    def _extract_person_detections(self, result: Any, frame_w: int, frame_h: int) -> List[Dict[str, Any]]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        detections: List[Dict[str, Any]] = []
        xyxy_values = boxes.xyxy.cpu().numpy() if getattr(boxes, "xyxy", None) is not None else []
        conf_values = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else np.zeros(len(xyxy_values))
        cls_values = boxes.cls.cpu().numpy() if getattr(boxes, "cls", None) is not None else np.zeros(len(xyxy_values))

        for bbox_raw, conf_raw, cls_raw in zip(xyxy_values, conf_values, cls_values):
            if int(cls_raw) != 0:
                continue
            confidence = float(conf_raw)
            if confidence < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = [float(v) for v in bbox_raw[:4]]
            box_w = max(0.0, x2 - x1)
            box_h = max(0.0, y2 - y1)
            if box_w <= 1.0 or box_h <= 1.0:
                continue

            height_ratio = box_h / max(1.0, float(frame_h))
            area_ratio = (box_w * box_h) / max(1.0, float(frame_w * frame_h))
            aspect_ratio = box_h / max(1e-6, box_w)
            if height_ratio < self.min_person_height_ratio:
                continue
            if area_ratio < self.min_person_area_ratio:
                continue
            if aspect_ratio < self.min_person_aspect_ratio:
                continue

            bbox = [x1, y1, x2, y2]
            detections.append(
                {
                    "bbox": bbox,
                    "center": self._xyxy_to_center(bbox),
                    "confidence": confidence,
                }
            )

        return detections

    def _assign_track_ids(self, detections: List[Dict[str, Any]], min_iou: float = 0.25) -> List[Dict[str, Any]]:
        self._frame_counter += 1
        assigned_tracks: List[Dict[str, Any]] = []
        used_track_ids = set()

        for det in sorted(detections, key=lambda d: float(d.get("confidence", 0.0)), reverse=True):
            bbox = det["bbox"]
            det_center = self._xyxy_to_center(bbox)
            best_track_id: Optional[int] = None
            best_score = -1.0

            for track_id, state in self._active_tracks.items():
                if track_id in used_track_ids:
                    continue
                iou = self._bbox_iou(state["bbox"], bbox)
                prev_center = state.get("center")
                if prev_center is None:
                    continue

                center_distance = math.hypot(
                    float(det_center[0]) - float(prev_center[0]),
                    float(det_center[1]) - float(prev_center[1]),
                )
                prev_bbox = state.get("bbox", [0.0, 0.0, 0.0, 0.0])
                prev_w = max(1.0, float(prev_bbox[2]) - float(prev_bbox[0]))
                prev_h = max(1.0, float(prev_bbox[3]) - float(prev_bbox[1]))
                dynamic_max_dist = max(35.0, min(180.0, math.hypot(prev_w, prev_h) * 0.9))

                if iou < min_iou and center_distance > dynamic_max_dist:
                    continue

                iou_score = max(0.0, min(1.0, float(iou)))
                dist_score = max(0.0, 1.0 - (center_distance / max(1e-6, dynamic_max_dist)))
                score = (0.7 * iou_score) + (0.3 * dist_score)
                if score > best_score:
                    best_score = score
                    best_track_id = track_id

            if best_track_id is None:
                best_track_id = self._next_track_id
                self._next_track_id += 1

            used_track_ids.add(best_track_id)
            center = self._xyxy_to_center(bbox)
            self._active_tracks[best_track_id] = {
                "bbox": bbox,
                "center": center,
                "last_seen": self._frame_counter,
            }
            assigned_tracks.append(
                {
                    "track_id": int(best_track_id),
                    "bbox": [float(v) for v in bbox],
                    "center": center,
                    "confidence": float(det.get("confidence", 0.0)),
                }
            )

        stale_ids = [
            track_id
            for track_id, state in self._active_tracks.items()
            if (self._frame_counter - int(state.get("last_seen", 0))) > self._max_lost_frames
        ]
        for track_id in stale_ids:
            self._active_tracks.pop(track_id, None)

        return assigned_tracks

    def detect(self, image) -> Dict[str, Any]:
        tracks = self.track(frame=image)
        confidences = [float(item["confidence"]) for item in tracks]
        return {
            "persons_detected": len(tracks),
            "persons": tracks,
            "confidences": confidences,
            "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "max_confidence": float(max(confidences)) if confidences else 0.0,
            "min_confidence": float(min(confidences)) if confidences else 0.0,
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
        del tracker
        t0 = time.time()
        try:
            model = self._ensure_model()
            results = model.predict(
                frame,
                conf=max(float(conf), self.confidence_threshold),
                iou=float(iou),
                imgsz=int(imgsz),
                augment=bool(augment),
                max_det=max(1, int(max_det)),
                classes=[0],
                device=self.device,
                verbose=False,
            )
            frame_h, frame_w = frame.shape[:2]
            detections = self._extract_person_detections(results[0] if results else None, frame_w, frame_h)
            tracks = self._assign_track_ids(detections)
            self.last_track_ok = True
            self.last_track_error = ""
            return tracks
        except Exception as exc:
            self.last_track_ok = False
            self.last_track_error = str(exc)
            return []
        finally:
            self.last_inference_seconds = max(0.0, time.time() - t0)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_path": self.model_path,
            "framework": "Ultralytics local YOLO",
            "confidence_threshold": self.confidence_threshold,
            "device": self.device,
            "task": "person_tracking",
        }
