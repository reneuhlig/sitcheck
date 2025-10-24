from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseDetector(ABC):
    """Abstrakte Basisklasse für alle KI-Detektoren"""
    
    def __init__(self, model_name: str, model_version: str = None):
        self.model_name = model_name
        self.model_version = model_version or "unknown"
        
    @abstractmethod
    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        Führt Detection auf einem Bild durch
        
        Args:
            image_path: Pfad zum Bild
            
        Returns:
            Dictionary mit Erkennungsergebnissen
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen über das Modell zurück
        
        Returns:
            Dictionary mit Modellinformationen
        """
        pass