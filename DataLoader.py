import time
from pathlib import Path
from typing import Optional, Generator
import threading
import cv2


class LiveImageLoader:
    def __init__(self, dir_x: str, dir_y: str, poll_interval: float = 0.5):
        self.dir_x = Path(dir_x)
        self.dir_y = Path(dir_y)
        self.poll_interval = poll_interval
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        self._stop_event = threading.Event()
        for d in (self.dir_x, self.dir_y):
            d.mkdir(parents=True, exist_ok=True)

    def _get_next_image_path(self, directory: Path) -> Optional[Path]:
        for file in directory.iterdir():
            if file.suffix.lower() in self.supported_formats and file.is_file():
                return file
        return None

    def _load_and_delete(self, file_path: Path):
        try:
            img = cv2.imread(str(file_path))
            if img is None:
                return None
            file_path.unlink(missing_ok=True)
            return img
        except Exception:
            return None

    def watch(self) -> Generator[tuple[str, any], None, None]:
        while not self._stop_event.is_set():
            for directory in (self.dir_x, self.dir_y):
                next_img = self._get_next_image_path(directory)
                if next_img:
                    img = self._load_and_delete(next_img)
                    if img is not None:
                        yield (directory.name, img)
            time.sleep(self.poll_interval)

    def stop(self):
        self._stop_event.set()


if __name__ == "__main__":
    loader = LiveImageLoader("input_x", "input_y", poll_interval=0.5)
    try:
        for source, img in loader.watch():
            print(source, img.shape)
    except KeyboardInterrupt:
        loader.stop()
