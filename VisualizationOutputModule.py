import cv2
from typing import Callable, Dict, List, Optional, Tuple

from TrajectoryEntryAnalysisModule import EntranceZoneConfig


class VisualizationOutputModule:
    """Rendert Live-Overlays und stellt interaktives Zone-Editing bereit."""

    def __init__(
        self,
        show_window: bool = True,
        window_name: str = "Library Entry Tracking",
        enable_zone_editor: bool = True,
        on_zone_changed: Optional[Callable[[EntranceZoneConfig], None]] = None,
    ):
        self.show_window = show_window
        self.window_name = window_name
        self.enable_zone_editor = enable_zone_editor
        self.on_zone_changed = on_zone_changed

        self._mouse_initialized = False
        self._frame_shape: Optional[Tuple[int, int, int]] = None
        self._zone_config: Optional[EntranceZoneConfig] = None
        self._line_click_stage = 0

    def draw(
        self,
        frame,
        tracks: List[Dict],
        zone_config: EntranceZoneConfig,
        occupancy: int,
        entries_total: int,
        exits_total: int,
        events_in_frame,
    ):
        output = frame.copy()
        self._frame_shape = output.shape
        self._zone_config = zone_config

        self._draw_zone(output, zone_config)

        for track in tracks:
            x1, y1, x2, y2 = track["bbox"]
            track_id = track.get("track_id")
            confidence = track.get("confidence", 0.0)
            color = (0, 200, 0)

            cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(
                output,
                f"ID {track_id} | {confidence:.2f}",
                (int(x1), max(20, int(y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        cv2.putText(output, f"Occupancy: {occupancy}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(output, f"Entries total: {entries_total}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(output, f"Exits total: {exits_total}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        if isinstance(events_in_frame, dict):
            entry_now = int(events_in_frame.get("entry", 0))
            exit_now = int(events_in_frame.get("exit", 0))
            cv2.putText(output, f"Frame events: +{entry_now} / -{exit_now}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        else:
            cv2.putText(output, f"Entries this frame: {events_in_frame}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.putText(
            output,
            "Editor: [L] line, [P] polygon, [D] direction, left-click add/set, right-click undo",
            (20, output.shape[0] - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            output,
            "[S] save now, [Q/ESC] quit",
            (20, output.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        return output

    def show(self, frame):
        if not self.show_window:
            return

        cv2.imshow(self.window_name, frame)
        if self.enable_zone_editor and not self._mouse_initialized:
            cv2.setMouseCallback(self.window_name, self._on_mouse)
            self._mouse_initialized = True

    @staticmethod
    def _to_norm(frame_shape: Tuple[int, int, int], x: int, y: int) -> Tuple[float, float]:
        frame_h, frame_w = frame_shape[:2]
        return max(0.0, min(1.0, x / frame_w)), max(0.0, min(1.0, y / frame_h))

    def _on_mouse(self, event, x, y, _flags, _param):
        if not self._zone_config or not self._frame_shape:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            nx, ny = self._to_norm(self._frame_shape, x, y)
            if self._zone_config.mode == "line":
                if self._line_click_stage == 0:
                    self._zone_config.line_p1 = (nx, ny)
                    self._line_click_stage = 1
                else:
                    self._zone_config.line_p2 = (nx, ny)
                    self._line_click_stage = 0
                self._notify_zone_changed()
            else:
                points = list(self._zone_config.polygon_points)
                points.append((nx, ny))
                self._zone_config.polygon_points = points
                self._notify_zone_changed()

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self._zone_config.mode == "polygon" and self._zone_config.polygon_points:
                points = list(self._zone_config.polygon_points)
                points.pop()
                self._zone_config.polygon_points = points
                self._notify_zone_changed()

    def handle_key(self, key: int):
        if not self._zone_config:
            return

        if key in (ord("l"), ord("L")):
            self._zone_config.mode = "line"
            self._notify_zone_changed()
        elif key in (ord("p"), ord("P")):
            self._zone_config.mode = "polygon"
            self._notify_zone_changed()
        elif key in (ord("d"), ord("D")):
            current = self._zone_config.line_entry_direction
            self._zone_config.line_entry_direction = (
                "positive_to_negative" if current == "negative_to_positive" else "negative_to_positive"
            )
            self._notify_zone_changed()
        elif key in (ord("s"), ord("S")):
            self._notify_zone_changed()

    def wait_key(self, wait_ms: int = 1) -> int:
        return cv2.waitKey(wait_ms) & 0xFF

    def _notify_zone_changed(self):
        if self.on_zone_changed and self._zone_config:
            self.on_zone_changed(self._zone_config)

    @staticmethod
    def should_quit(key: int) -> bool:
        return key in (ord("q"), ord("Q"), 27)

    def close(self):
        if self.show_window:
            cv2.destroyAllWindows()

    def _draw_zone(self, output, zone_config: EntranceZoneConfig):
        frame_h, frame_w = output.shape[:2]

        if zone_config.mode == "dual_polygon":
            entry_points = [(int(px * frame_w), int(py * frame_h)) for px, py in zone_config.entry_polygon_points]
            exit_points = [(int(px * frame_w), int(py * frame_h)) for px, py in zone_config.exit_polygon_points]

            if len(entry_points) >= 2:
                for idx in range(len(entry_points) - 1):
                    cv2.line(output, entry_points[idx], entry_points[idx + 1], (0, 200, 0), 2)
            if len(entry_points) >= 3:
                cv2.line(output, entry_points[-1], entry_points[0], (0, 200, 0), 2)
            for p in entry_points:
                cv2.circle(output, p, 4, (0, 255, 0), -1)

            if len(exit_points) >= 2:
                for idx in range(len(exit_points) - 1):
                    cv2.line(output, exit_points[idx], exit_points[idx + 1], (0, 0, 255), 2)
            if len(exit_points) >= 3:
                cv2.line(output, exit_points[-1], exit_points[0], (0, 0, 255), 2)
            for p in exit_points:
                cv2.circle(output, p, 4, (0, 0, 255), -1)

            cv2.putText(output, "Mode: dual_polygon | Green=Entry | Red=Exit", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)
            return

        if zone_config.mode == "line":
            p1 = (int(zone_config.line_p1[0] * frame_w), int(zone_config.line_p1[1] * frame_h))
            p2 = (int(zone_config.line_p2[0] * frame_w), int(zone_config.line_p2[1] * frame_h))
            cv2.line(output, p1, p2, (0, 255, 255), 2)
            cv2.circle(output, p1, 5, (255, 255, 0), -1)
            cv2.circle(output, p2, 5, (255, 255, 0), -1)
            cv2.putText(
                output,
                f"Zone: line | direction: {zone_config.line_entry_direction}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
        else:
            points = [(int(px * frame_w), int(py * frame_h)) for px, py in zone_config.polygon_points]
            if len(points) >= 2:
                for idx in range(len(points) - 1):
                    cv2.line(output, points[idx], points[idx + 1], (255, 100, 0), 2)
            if len(points) >= 3:
                cv2.line(output, points[-1], points[0], (255, 100, 0), 2)
            for p in points:
                cv2.circle(output, p, 4, (255, 100, 0), -1)
            cv2.putText(output, "Zone: polygon (outside->inside)", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
