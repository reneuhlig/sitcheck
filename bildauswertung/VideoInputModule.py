import cv2
import time
import os
from typing import Optional, Tuple, Any


class VideoInputModule:
    """Verwaltet einen Live-Video-Input (Webcam/RTSP/Datei/Stream)."""

    def __init__(
        self,
        source: str,
        fallback_source: str | int | None = None,
        reconnect_delay: float = 1.0,
        max_retries: int = 0,
        hwaccel: str = "auto",
        youtube_cookies_from_browser: str = "",
        youtube_cookiefile: str = "",
        youtube_format: str = "best[ext=mp4]/best",
        youtube_player_client: str = "android",
    ):
        self.raw_source = source
        self.source = self._parse_source(source)
        self.fallback_source = None if fallback_source in (None, "") else self._parse_source(str(fallback_source))
        self.is_youtube_source = self._is_youtube_url(str(source))
        self.reconnect_delay = reconnect_delay
        self.max_retries = max_retries
        self.hwaccel = str(hwaccel or "auto").lower()
        self.youtube_cookies_from_browser = str(youtube_cookies_from_browser or "").strip()
        self.youtube_cookiefile = str(youtube_cookiefile or "").strip()
        self.youtube_format = str(youtube_format or "best[ext=mp4]/best")
        self.youtube_player_client = str(youtube_player_client or "android").strip() or "android"
        self.capture: Optional[cv2.VideoCapture] = None
        self.active_source: Any = self.source
        self._retries = 0
        self._last_youtube_resolve_ts = 0.0
        self._youtube_resolve_failures = 0

    @staticmethod
    def _parse_source(source: str):
        stripped = str(source).strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped

    @staticmethod
    def _is_youtube_url(url: str) -> bool:
        return "youtube.com/watch" in url or "youtu.be/" in url

    def _resolve_youtube_stream(self, url: str) -> Optional[str]:
        """
        Löst YouTube URL in direkte Video-Stream-URL auf.
        Erfordert `yt-dlp`; bei Fehler wird None zurückgegeben.
        """
        try:
            from yt_dlp import YoutubeDL

            base_options = {
                "quiet": True,
                "no_warnings": True,
                "format": self.youtube_format,
                "noplaylist": True,
                "extractor_args": {
                    "youtube": {
                        "player_client": [self.youtube_player_client, "web"],
                    }
                },
            }

            option_candidates = [dict(base_options)]

            if self.youtube_cookies_from_browser:
                browser_parts = tuple(part for part in self.youtube_cookies_from_browser.split(":") if part)
                if browser_parts:
                    options_with_browser = dict(base_options)
                    options_with_browser["cookiesfrombrowser"] = browser_parts
                    option_candidates.append(options_with_browser)
            else:
                for browser in ("firefox", "chrome", "chromium", "brave"):
                    options_with_browser = dict(base_options)
                    options_with_browser["cookiesfrombrowser"] = (browser,)
                    option_candidates.append(options_with_browser)

            if self.youtube_cookiefile and os.path.exists(self.youtube_cookiefile):
                options_with_cookiefile = dict(base_options)
                options_with_cookiefile["cookiefile"] = self.youtube_cookiefile
                option_candidates.append(options_with_cookiefile)

            for options in option_candidates:
                try:
                    with YoutubeDL(options) as ydl:
                        info = ydl.extract_info(url, download=False)
                        direct_url = info.get("url") if isinstance(info, dict) else None
                        if direct_url:
                            return str(direct_url)

                        requested_formats = info.get("requested_formats", []) if isinstance(info, dict) else []
                        for fmt in requested_formats:
                            fmt_url = fmt.get("url") if isinstance(fmt, dict) else None
                            if fmt_url:
                                return str(fmt_url)
                except Exception:
                    continue

            return None
        except Exception:
            return None

    def open(self) -> bool:
        source_to_open = self.source
        if self.is_youtube_source:
            now = time.monotonic()
            retry_backoff = min(30.0, max(1.0, float(self.reconnect_delay)) * (2 ** min(self._youtube_resolve_failures, 4)))
            if self._youtube_resolve_failures > 0 and (now - self._last_youtube_resolve_ts) < retry_backoff:
                return False

            self._last_youtube_resolve_ts = now
            resolved_source = self._resolve_youtube_stream(str(self.raw_source))
            if not resolved_source:
                self._youtube_resolve_failures += 1
                if self.fallback_source is None:
                    return False
                source_to_open = self.fallback_source
            else:
                source_to_open = resolved_source
                self.source = resolved_source
                self._youtube_resolve_failures = 0

        if isinstance(source_to_open, int):
            self.capture = cv2.VideoCapture(source_to_open)
        else:
            self.capture = cv2.VideoCapture(source_to_open, cv2.CAP_FFMPEG)
            if not (self.capture and self.capture.isOpened()):
                try:
                    if self.capture:
                        self.capture.release()
                except Exception:
                    pass
                self.capture = cv2.VideoCapture(source_to_open)

        if self.capture and self.capture.isOpened():
            self.active_source = source_to_open
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._configure_hwaccel()
        return bool(self.capture and self.capture.isOpened())

    def _configure_hwaccel(self):
        if self.hwaccel not in {"auto", "vaapi", "cuda", "any"}:
            return
        if not self.capture:
            return

        cap_prop_hwaccel = getattr(cv2, "CAP_PROP_HW_ACCELERATION", None)
        if cap_prop_hwaccel is None:
            return

        accel_any = getattr(cv2, "VIDEO_ACCELERATION_ANY", 1)
        accel_vaapi = getattr(cv2, "VIDEO_ACCELERATION_VAAPI", accel_any)
        accel_cuda = getattr(cv2, "VIDEO_ACCELERATION_CUDA", accel_any)

        if self.hwaccel == "vaapi":
            self.capture.set(cap_prop_hwaccel, accel_vaapi)
            return
        if self.hwaccel == "cuda":
            self.capture.set(cap_prop_hwaccel, accel_cuda)
            return

        self.capture.set(cap_prop_hwaccel, accel_any)

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
