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
        self.preprocess_enabled = preprocess_enabled
        self.preprocess_upscale = preprocess_upscale
        self.preprocess_clahe_clip = preprocess_clahe_clip
        self.preprocess_denoise = preprocess_denoise

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
        clahe = cv2.createCLAHE(clipLimit=max(1.0, float(self.preprocess_clahe_clip)), tileGridSize=(8, 8))
        y = clahe.apply(y)
        ycrcb = cv2.merge([y, cr, cb])
        processed = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

        if self.preprocess_denoise:
            processed = cv2.fastNlMeansDenoisingColored(processed, None, 5, 5, 7, 21)

        return processed

    def track(self, frame) -> List[Dict[str, Any]]:
        model_input = self._preprocess_frame(frame) if self.preprocess_enabled else frame
        return self.detector.track(
            frame=model_input,
            tracker=self.tracker_config,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
        )
