import time
from pathlib import Path
from typing import Optional, Generator, Tuple, List
import threading
import cv2


class LiveImageLoader:
    def __init__(self, dir_x: str, dir_y: str, poll_interval: float = 0.5):
        self.dir_x = Path(dir_x)
        self.dir_y = Path(dir_y)
        self.poll_interval = poll_interval
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        self._stop_event = threading.Event()
        
        # Queue für geladene Bilder mit Pfad für späteres Löschen
        self._pending_deletions = []
        
        for d in (self.dir_x, self.dir_y):
            d.mkdir(parents=True, exist_ok=True)

    def _get_all_image_paths(self, directory: Path) -> List[Path]:
        """Holt ALLE Bilddateien aus einem Verzeichnis (sortiert nach Name)"""
        files = []
        for file in directory.iterdir():
            if file.suffix.lower() in self.supported_formats and file.is_file():
                files.append(file)
        return sorted(files)  # Sortierung für konsistente Reihenfolge

    def _is_file_stable(self, file_path: Path, checks: int = 2, interval: float = 0.1) -> bool:
        """Prüft ob Datei vollständig übertragen wurde (stabile Größe)"""
        try:
            size1 = file_path.stat().st_size
            for _ in range(checks):
                time.sleep(interval)
                size2 = file_path.stat().st_size
                if size1 != size2:
                    return False
                size1 = size2
            return True
        except Exception:
            return False

    def _load_image(self, file_path: Path) -> Tuple[Optional[any], Path]:
        """
        Lädt ein Bild in den Speicher, OHNE es zu löschen
        
        Returns:
            Tuple (Bild oder None, Dateipfad)
        """
        try:
            # Warte bis Datei vollständig übertragen wurde
            if not self._is_file_stable(file_path):
                return None, file_path
            
            img = cv2.imread(str(file_path))
            if img is None:
                # Fehlerhaftes Bild direkt löschen
                file_path.unlink(missing_ok=True)
                return None, None
            
            return img, file_path
        except Exception:
            return None, file_path

    def confirm_processed(self, file_path: Path):
        """
        Löscht eine Datei nachdem sie erfolgreich verarbeitet wurde
        
        Args:
            file_path: Pfad zur Datei die gelöscht werden soll
        """
        try:
            if file_path and file_path.exists():
                file_path.unlink(missing_ok=True)
        except Exception as e:
            pass  # Fehler beim Löschen ignorieren

    def watch(self) -> Generator[tuple[str, any, Path], None, None]:
        """
        Überwacht Ordner und gibt Bilder zurück
        
        Yields:
            Tuple (source_name, image, file_path)
            - source_name: Name des Quellordners
            - image: OpenCV Bild-Array
            - file_path: Pfad zur Datei (für späteres Löschen)
        """
        while not self._stop_event.is_set():
            for directory in (self.dir_x, self.dir_y):
                # Hole ALLE Bilder aus diesem Ordner
                image_paths = self._get_all_image_paths(directory)
                
                # Verarbeite sequentiell
                for img_path in image_paths:
                    img, path = self._load_image(img_path)
                    if img is not None and path is not None:
                        # Gebe Bild UND Pfad zurück (Pfad für späteres Löschen)
                        yield (directory.name, img, path)
            
            time.sleep(self.poll_interval)

    def stop(self):
        self._stop_event.set()


if __name__ == "__main__":
    loader = LiveImageLoader("input_x", "input_y", poll_interval=0.5)
    try:
        for source, img, file_path in loader.watch():
            print(f"{source}: {img.shape} - {file_path.name}")
            # Simuliere Verarbeitung
            time.sleep(0.1)
            # Nach erfolgreicher "Verarbeitung" löschen
            loader.confirm_processed(file_path)
    except KeyboardInterrupt:
        loader.stop()