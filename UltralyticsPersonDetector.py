from ultralytics import YOLO
import cv2
import numpy as np
from typing import Dict, Any, List
from BaseDetector import BaseDetector


class UltralyticsPersonDetector(BaseDetector):
    """Ultralytics YOLO Detektor für Personenerkennung"""
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Initialisiert den Ultralytics Detektor
        
        Args:
            model_path: Pfad zum YOLO Modell
            confidence_threshold: Mindest-Konfidenz für Detections
        """
        super().__init__("Ultralytics YOLO", "v8")
        self.confidence_threshold = confidence_threshold
        self.model = YOLO(model_path)
        
        # Person class ID in COCO dataset is 0
        self.person_class_id = 0
        
    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        Erkennt Personen in einem Bild
        
        Args:
            image_path: Pfad zum Bild
            
        Returns:
            Dictionary mit Erkennungsergebnissen
        """
        try:
            # Bild laden und Detection durchführen
            results = self.model(image_path, verbose=False)
            
            # Personen extrahieren
            persons = []
            confidences = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Nur Personen (class_id = 0) berücksichtigen
                        if int(box.cls) == self.person_class_id:
                            confidence = float(box.conf)
                            if confidence >= self.confidence_threshold:
                                persons.append({
                                    'bbox': box.xyxy[0].cpu().numpy().tolist(),
                                    'confidence': confidence,
                                    'class': 'person'
                                })
                                confidences.append(confidence)
            
            # Ergebnis zusammenstellen
            return {
                'persons_detected': len(persons),
                'persons': persons,
                'confidences': confidences,
                'avg_confidence': np.mean(confidences) if confidences else 0.0,
                'max_confidence': max(confidences) if confidences else 0.0,
                'min_confidence': min(confidences) if confidences else 0.0,
                'uncertain': any(c < 0.7 for c in confidences) if confidences else False,
                'model_output': {
                    'total_detections': len(persons),
                    'high_confidence_count': len([c for c in confidences if c >= 0.8]),
                    'medium_confidence_count': len([c for c in confidences if 0.5 <= c < 0.8]),
                    'low_confidence_count': len([c for c in confidences if c < 0.5])
                }
            }
            
        except Exception as e:
            return {
                'persons_detected': 0,
                'persons': [],
                'confidences': [],
                'avg_confidence': 0.0,
                'max_confidence': 0.0,
                'min_confidence': 0.0,
                'uncertain': True,
                'error': str(e),
                'model_output': {}
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Gibt Modellinformationen zurück"""
        return {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'framework': 'Ultralytics',
            'confidence_threshold': self.confidence_threshold,
            'task': 'person_detection',
            'input_size': 'variable',
            'classes': ['person']
        }