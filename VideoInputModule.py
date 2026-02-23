import cv2
import time
from typing import Optional, Tuple, Any


class VideoInputModule:
    """Verwaltet einen Live-Video-Input (Webcam/RTSP/Datei/Stream)."""

    def __init__(
        self,
        source: str,
        reconnect_delay: float = 1.0,
        max_retries: int = 0,
    ):
        self.raw_source = source
        self.source = self._parse_source(source)
        self.reconnect_delay = reconnect_delay
        self.max_retries = max_retries
        self.capture: Optional[cv2.VideoCapture] = None
        self._retries = 0

    @staticmethod
    def _parse_source(source: str):
        stripped = str(source).strip()
        if stripped.isdigit():
            return int(stripped)
        if "youtube.com/watch" in stripped or "youtu.be/" in stripped:
            return VideoInputModule._resolve_youtube_stream(stripped)
        return stripped

    @staticmethod
    def _resolve_youtube_stream(url: str) -> Any:
        """
        Löst YouTube URL in direkte Video-Stream-URL auf.
        Erfordert `yt-dlp`; bei Fehler wird die Original-URL zurückgegeben.
        """
        try:
            from yt_dlp import YoutubeDL

            options = {
                "quiet": True,
                "no_warnings": True,
                "format": "best[ext=mp4]/best",
            }
            with YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=False)
                return info.get("url", url)
        except Exception:
            # Fallback: OpenCV versucht trotzdem zu öffnen.
            return url

    def open(self) -> bool:
        self.capture = cv2.VideoCapture(self.source)
        return bool(self.capture and self.capture.isOpened())

    def read(self) -> Tuple[bool, Optional[any]]:
        if not self.capture or not self.capture.isOpened():
            if not self._try_reconnect():
                return False, None

        ok, frame = self.capture.read()
        if ok:
            self._retries = 0
            return True, frame

        if not self._try_reconnect():
            return False, None

        return self.capture.read()

    def _try_reconnect(self) -> bool:
        if self.max_retries > 0 and self._retries >= self.max_retries:
            return False

        self.release()
        time.sleep(self.reconnect_delay)
        self._retries += 1
        return self.open()

    def release(self):
        if self.capture:
            self.capture.release()
            self.capture = None
