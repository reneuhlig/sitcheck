import time
from typing import Dict, Any, List, Optional, Tuple
import math

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from BaseDetector import BaseDetector


class UltralyticsPersonDetector(BaseDetector):
    """Detector ueber Ultralytics Inference API statt lokalem Modell."""
    
    def __init__(
        self,
        api_url: str,
        api_key: str,
        confidence_threshold: float = 0.5,
        device: str = "cpu",
        request_timeout_seconds: float = 10.0,
        failure_cooldown_seconds: float = 2.0,
        max_api_fps: float = 0.0,
        jpeg_quality: int = 85,
        min_person_height_ratio: float = 0.10,
        min_person_area_ratio: float = 0.010,
        min_person_aspect_ratio: float = 1.00,
        allow_classless_person_detections: bool = False,
    ):
        """
        Initialisiert den Ultralytics Detektor
        
        Args:
            api_url: Deployment URL der Inference API
            api_key: API-Key fuer Bearer-Auth
            confidence_threshold: Mindest-Konfidenz fuer Detections
            device: Bleibt aus Kompatibilitaetsgruenden erhalten
            request_timeout_seconds: HTTP Timeout pro Request
            failure_cooldown_seconds: Fast-Fail-Fenster nach Requestfehlern
            max_api_fps: Begrenzung der API-Request-Rate (0 = unbegrenzt)
            jpeg_quality: JPEG-Qualitaet fuer API Upload (1-100)
            min_person_height_ratio: Mindesthoehe der Person-Box relativ zur Framehoehe
            min_person_area_ratio: Mindestflaeche der Person-Box relativ zur Frameflaeche
            min_person_aspect_ratio: Mindestverhaeltnis Hoehe/Breite fuer Personenbox
            allow_classless_person_detections: Akzeptiert classless Detections als Person
        """
        super().__init__("Ultralytics API", "v1")
        self.confidence_threshold = confidence_threshold
        normalized_device = str(device).strip().lower()
        self.device = "cpu" if normalized_device in {"", "auto"} else normalized_device
        self.api_url = str(api_url).strip()
        self.api_key = str(api_key).strip()
        self.request_timeout_seconds = max(1.0, float(request_timeout_seconds))
        self.failure_cooldown_seconds = max(0.0, float(failure_cooldown_seconds))
        self.max_api_fps = max(0.0, float(max_api_fps))
        self._min_api_interval_seconds = (1.0 / self.max_api_fps) if self.max_api_fps > 0 else 0.0
        self.jpeg_quality = max(60, min(95, int(jpeg_quality)))
        self.min_person_height_ratio = max(0.0, min(1.0, float(min_person_height_ratio)))
        self.min_person_area_ratio = max(0.0, min(1.0, float(min_person_area_ratio)))
        self.min_person_aspect_ratio = max(0.0, float(min_person_aspect_ratio))
        self.allow_classless_person_detections = bool(allow_classless_person_detections)
        if not self.api_url:
            raise ValueError("tracking.api_url darf nicht leer sein")
        if not self.api_key:
            raise ValueError("tracking.api_key darf nicht leer sein")

        print(f"[INFO] Nutze Inference API: {self.api_url}", flush=True)
        self.person_class_id = 0  # Person class ID in COCO dataset
        self.last_track_ok = True
        self.last_track_error = ""
        self._last_track_error_log_ts = 0.0
        self._active_tracks: Dict[int, Dict[str, Any]] = {}
        self._next_track_id = 1
        self._max_lost_frames = 30
        self._frame_counter = 0
        self._failure_cooldown_until_ts = 0.0
        self._last_api_request_ts = 0.0

        # Keep-Alive Session reduziert TLS/Handshake-Overhead deutlich.
        self._http = requests.Session()
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=0)
        self._http.mount("http://", adapter)
        self._http.mount("https://", adapter)

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
        denom = (area_a + area_b - inter_area)
        if denom <= 0.0:
            return 0.0
        return inter_area / denom

    def _call_predict_api(self, image, conf: float, iou: float, imgsz: int) -> List[Dict[str, Any]]:
        now_ts = time.time()
        if self._failure_cooldown_until_ts > now_ts:
            remaining = max(0.0, self._failure_cooldown_until_ts - now_ts)
            raise RuntimeError(
                f"Inference API temporaer im Cooldown ({remaining:.1f}s verbleibend) nach vorherigem Fehler"
            )

        if self._min_api_interval_seconds > 0.0:
            elapsed = now_ts - self._last_api_request_ts
            sleep_for = self._min_api_interval_seconds - elapsed
            if sleep_for > 0:
                time.sleep(min(sleep_for, 0.25))

        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise RuntimeError("Frame konnte nicht als JPEG encodiert werden")

        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "conf": max(float(conf), self.confidence_threshold),
            "iou": float(iou),
            "imgsz": int(imgsz),
        }
        files = {"file": ("frame.jpg", encoded.tobytes(), "image/jpeg")}
        timeout = (min(2.0, self.request_timeout_seconds), self.request_timeout_seconds)
        try:
            self._last_api_request_ts = time.time()
            response = self._http.post(
                self.api_url,
                headers=headers,
                data=payload,
                files=files,
                timeout=timeout,
            )
            response.raise_for_status()
            body = response.json()
        except requests.exceptions.Timeout as exc:
            self._failure_cooldown_until_ts = time.time() + self.failure_cooldown_seconds
            raise RuntimeError(
                f"Inference API Timeout nach {self.request_timeout_seconds:.1f}s"
            ) from exc
        except requests.exceptions.RequestException as exc:
            self._failure_cooldown_until_ts = time.time() + self.failure_cooldown_seconds
            raise RuntimeError(f"Inference API Fehler: {exc}") from exc

        self._failure_cooldown_until_ts = 0.0
        frame_h, frame_w = image.shape[:2]
        return self._extract_person_detections(body, frame_w=frame_w, frame_h=frame_h)

    def _extract_person_detections(self, body: Any, frame_w: int, frame_h: int) -> List[Dict[str, Any]]:
        if isinstance(body, dict) and isinstance(body.get("images"), list):
            candidates = []
            for image_item in body.get("images", []):
                if not isinstance(image_item, dict):
                    continue
                image_results = image_item.get("results")
                if isinstance(image_results, list):
                    candidates.extend(image_results)
        elif isinstance(body, dict):
            candidates = body.get("predictions")
            if candidates is None:
                candidates = body.get("results")
            if candidates is None:
                candidates = body.get("boxes")
            if candidates is None:
                candidates = body.get("detections")
            if candidates is None:
                candidates = []
        elif isinstance(body, list):
            candidates = body
        else:
            candidates = []

        # Manche Endpunkte liefern `results=[{"boxes": [...]}]`.
        if isinstance(candidates, list) and len(candidates) == 1 and isinstance(candidates[0], dict):
            nested_boxes = candidates[0].get("boxes")
            if isinstance(nested_boxes, list):
                candidates = nested_boxes

        detections: List[Dict[str, Any]] = []
        for item in candidates:
            if not isinstance(item, dict):
                continue

            class_id = item.get("class", item.get("class_id", item.get("cls", None)))
            class_name = str(item.get("name", item.get("class_name", ""))).strip().lower()
            is_person_label = (
                str(class_id).strip() in {"0", "person"}
                or class_name == "person"
            )
            is_classless = class_id is None and class_name == ""
            is_person = is_person_label or (self.allow_classless_person_detections and is_classless)
            if not is_person:
                continue

            bbox = item.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            else:
                box_dict = item.get("box") if isinstance(item.get("box"), dict) else {}
                x1 = item.get("x1", box_dict.get("x1"))
                y1 = item.get("y1", box_dict.get("y1"))
                x2 = item.get("x2", box_dict.get("x2"))
                y2 = item.get("y2", box_dict.get("y2"))
                if None in {x1, y1, x2, y2}:
                    x_center = item.get("x")
                    y_center = item.get("y")
                    width = item.get("width", item.get("w"))
                    height = item.get("height", item.get("h"))
                    if None in {x_center, y_center, width, height}:
                        continue
                    x1 = float(x_center) - (float(width) / 2.0)
                    y1 = float(y_center) - (float(height) / 2.0)
                    x2 = float(x_center) + (float(width) / 2.0)
                    y2 = float(y_center) + (float(height) / 2.0)
                else:
                    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

            conf = item.get("confidence", item.get("conf", item.get("score", 0.0)))
            confidence = float(conf)
            if confidence < self.confidence_threshold:
                continue

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

            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "center": self._xyxy_to_center([x1, y1, x2, y2]),
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
        """
        Erkennt Personen in einem Bild
        
        Args:
            image: OpenCV Bildobjekt (numpy array)
            
        Returns:
            Dictionary mit Erkennungsergebnissen
        """
        try:
            persons = self._call_predict_api(
                image=image,
                conf=self.confidence_threshold,
                iou=0.7,
                imgsz=640,
            )
            confidences = [float(item["confidence"]) for item in persons]
            
            # Ergebnis zusammenstellen
            return {
                'persons_detected': len(persons),
                'persons': persons,
                'confidences': confidences,
                'avg_confidence': float(np.mean(confidences)) if confidences else 0.0,
                'max_confidence': float(max(confidences)) if confidences else 0.0,
                'min_confidence': float(min(confidences)) if confidences else 0.0
            }
            
        except Exception as e:
            return {
                'persons_detected': 0,
                'persons': [],
                'confidences': [],
                'avg_confidence': 0.0,
                'max_confidence': 0.0,
                'min_confidence': 0.0,
                'error': str(e)
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
        """
        Fuehrt YOLO-Tracking mit persistenten IDs aus.

        Verwendet explizit die dokumentierte API `model.track(...)`.
        """
        try:
            detections = self._call_predict_api(
                image=frame,
                conf=max(conf, self.confidence_threshold),
                iou=iou,
                imgsz=imgsz,
            )
            tracks = self._assign_track_ids(detections)

            self.last_track_ok = True
            self.last_track_error = ""
            return tracks

        except Exception as exc:
            self.last_track_ok = False
            self.last_track_error = str(exc)
            now_ts = time.time()
            if now_ts - self._last_track_error_log_ts >= 5.0:
                print(f"[TRACK-ERROR] {exc}", flush=True)
                self._last_track_error_log_ts = now_ts
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Gibt Modellinformationen zurueck
        
        Returns:
            Dictionary mit Modellinformationen
        """
        return {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'model_path': '',
            'framework': 'Ultralytics Inference API',
            'confidence_threshold': self.confidence_threshold,
            'device': self.device,
            'api_url': self.api_url,
            'task': 'person_tracking'
        }
