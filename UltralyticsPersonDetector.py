from ultralytics import YOLO
import numpy as np
from typing import Dict, Any, List
from BaseDetector import BaseDetector


def _as_numpy(value):
    if value is None:
        return np.array([])
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


class UltralyticsPersonDetector(BaseDetector):
    """Ultralytics YOLO Detektor fuer Personenerkennung und Tracking."""
    
    def __init__(self, model_path: str = "yolo26n.pt", confidence_threshold: float = 0.5, device: str = "cpu"):
        """
        Initialisiert den Ultralytics Detektor
        
        Args:
            model_path: Pfad zum YOLO Modell
            confidence_threshold: Mindest-Konfidenz fuer Detections
            device: Inferenz-Device (z.B. 'cpu', 'cuda:0')
        """
        super().__init__("Ultralytics YOLO", "v26-track")
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.person_class_id = 0  # Person class ID in COCO dataset
        
    def detect(self, image) -> Dict[str, Any]:
        """
        Erkennt Personen in einem Bild
        
        Args:
            image: OpenCV Bildobjekt (numpy array)
            
        Returns:
            Dictionary mit Erkennungsergebnissen
        """
        try:
            # Detection durchfuehren (klassische Einzelbild-Erkennung)
            results = self.model(image, verbose=False, device=self.device)
            
            # Personen extrahieren
            persons = []
            confidences = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Nur Personen (class_id = 0) beruecksichtigen
                        if int(box.cls) == self.person_class_id:
                            confidence = float(box.conf)
                            if confidence >= self.confidence_threshold:
                                persons.append({
                                    'bbox': box.xyxy[0].cpu().numpy().tolist(),
                                    'confidence': confidence
                                })
                                confidences.append(confidence)
            
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
    ) -> List[Dict[str, Any]]:
        """
        Fuehrt YOLO-Tracking mit persistenten IDs aus.

        Verwendet explizit die dokumentierte API `model.track(...)`.
        """
        try:
            results = self.model.track(
                source=frame,
                persist=True,
                classes=[self.person_class_id],
                tracker=tracker,
                conf=max(conf, self.confidence_threshold),
                iou=iou,
                imgsz=imgsz,
                device=self.device,
                verbose=False,
            )

            tracks = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                xyxy = _as_numpy(boxes.xyxy)
                confs = _as_numpy(boxes.conf)
                ids = _as_numpy(boxes.id)
                classes = _as_numpy(boxes.cls)

                for idx, bbox in enumerate(xyxy):
                    class_id = int(classes[idx]) if len(classes) > idx else self.person_class_id
                    if class_id != self.person_class_id:
                        continue

                    track_id = int(ids[idx]) if len(ids) > idx else None
                    if track_id is None:
                        continue

                    x1, y1, x2, y2 = [float(v) for v in bbox.tolist()]
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    tracks.append(
                        {
                            "track_id": track_id,
                            "bbox": [x1, y1, x2, y2],
                            "center": (cx, cy),
                            "confidence": float(confs[idx]) if len(confs) > idx else 0.0,
                        }
                    )

            return tracks

        except Exception:
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
            'model_path': self.model_path,
            'framework': 'Ultralytics',
            'confidence_threshold': self.confidence_threshold,
            'device': self.device,
            'task': 'person_tracking'
        }