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
        track_event_overlay: Optional[Dict[int, Dict]] = None,
        analysis_roi: Optional[Dict] = None,
    ):
        output = frame.copy()
        self._frame_shape = output.shape
        self._zone_config = zone_config

        self._draw_zone(output, zone_config)
        self._draw_analysis_roi(output, analysis_roi)

        for track in tracks:
            x1, y1, x2, y2 = track["bbox"]
            track_id = track.get("track_id")
            confidence = track.get("confidence", 0.0)
            is_stale = bool(track.get("is_stale", False))

            event_tag = None
            if track_event_overlay and track_id is not None:
                event_payload = track_event_overlay.get(int(track_id))
                if event_payload:
                    event_tag = str(event_payload.get("type", "")).upper()

            state_tag = event_tag if event_tag in {"ENTRY", "EXIT"} else "PASSING"
            if state_tag == "ENTRY":
                color = (0, 255, 0)
                badge_bg = (0, 110, 0)
            elif state_tag == "EXIT":
                color = (0, 0, 255)
                badge_bg = (0, 0, 140)
            else:
                color = (0, 215, 255) if not is_stale else (80, 160, 220)
                badge_bg = (130, 80, 0) if not is_stale else (90, 90, 90)

            cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            trail = track.get("trail", [])
            if len(trail) >= 2:
                trail_points = [(int(px), int(py)) for px, py in trail]
                for idx in range(1, len(trail_points)):
                    cv2.line(output, trail_points[idx - 1], trail_points[idx], (180, 180, 180), 2)

            center = track.get("center")
            motion_vector = track.get("motion_vector", (0.0, 0.0))
            motion_magnitude = float(track.get("motion_magnitude", 0.0))
            motion_direction = str(track.get("motion_direction", "still"))
            if center is not None and motion_magnitude >= 2.0 and len(motion_vector) == 2:
                start_point = (int(center[0]), int(center[1]))
                arrow_scale = 4.0
                end_point = (
                    int(center[0] + float(motion_vector[0]) * arrow_scale),
                    int(center[1] + float(motion_vector[1]) * arrow_scale),
                )
                cv2.arrowedLine(output, start_point, end_point, (0, 255, 255), 2, tipLength=0.35)

            label_text = f"ID {track_id} | {state_tag} | {confidence:.2f}"
            label_w = max(160, min(260, 8 * len(label_text)))
            label_x = int(x1)
            label_y = max(4, int(y1) - 20)
            cv2.rectangle(output, (label_x, label_y), (label_x + label_w, label_y + 18), (0, 0, 0), -1)
            cv2.putText(
                output,
                label_text,
                (label_x + 4, label_y + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
            )

            cv2.putText(
                output,
                motion_direction,
                (int(x1) + 4, max(22, int(y1) - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (220, 220, 220),
                1,
            )

            badge_text = "ENTRY +" if state_tag == "ENTRY" else ("EXIT -" if state_tag == "EXIT" else "PASSING")
            badge_w = 86 if state_tag == "PASSING" else 72
            text_x = int(x1)
            text_y = min(output.shape[0] - 8, int(y2) + 18)
            cv2.rectangle(output, (text_x, text_y - 14), (text_x + badge_w, text_y + 4), badge_bg, -1)
            cv2.putText(output, badge_text, (text_x + 4, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        legend_x = output.shape[1] - 220
        legend_y = 20
        cv2.rectangle(output, (legend_x - 10, legend_y - 12), (output.shape[1] - 10, legend_y + 78), (0, 0, 0), -1)
        cv2.putText(output, "Track Status", (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.rectangle(output, (legend_x, legend_y + 10), (legend_x + 12, legend_y + 22), (0, 255, 0), -1)
        cv2.putText(output, "ENTRY", (legend_x + 18, legend_y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.rectangle(output, (legend_x, legend_y + 30), (legend_x + 12, legend_y + 42), (0, 0, 255), -1)
        cv2.putText(output, "EXIT", (legend_x + 18, legend_y + 41), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.rectangle(output, (legend_x, legend_y + 50), (legend_x + 12, legend_y + 62), (0, 215, 255), -1)
        cv2.putText(output, "PASSING", (legend_x + 18, legend_y + 61), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

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

    @staticmethod
    def _draw_analysis_roi(output, analysis_roi: Optional[Dict]):
        if not analysis_roi or not bool(analysis_roi.get("enabled", False)):
            return

        frame_h, frame_w = output.shape[:2]
        mode = str(analysis_roi.get("mode", "rect")).lower()
        if mode == "polygon":
            points_norm = analysis_roi.get("polygon_points", []) or []
            points = []
            for pt in points_norm:
                if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                    continue
                px = max(0.0, min(1.0, float(pt[0])))
                py = max(0.0, min(1.0, float(pt[1])))
                points.append((int(px * frame_w), int(py * frame_h)))

            if len(points) >= 2:
                for idx in range(len(points) - 1):
                    cv2.line(output, points[idx], points[idx + 1], (255, 0, 255), 2)
            if len(points) >= 3:
                cv2.line(output, points[-1], points[0], (255, 0, 255), 2)
            for point in points:
                cv2.circle(output, point, 4, (255, 0, 255), -1)
            if points:
                cv2.putText(output, "Analysis ROI (polygon)", (points[0][0], max(20, points[0][1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
            return

        x_min = max(0.0, min(1.0, float(analysis_roi.get("x_min", 0.0))))
        y_min = max(0.0, min(1.0, float(analysis_roi.get("y_min", 0.0))))
        x_max = max(0.0, min(1.0, float(analysis_roi.get("x_max", 1.0))))
        y_max = max(0.0, min(1.0, float(analysis_roi.get("y_max", 1.0))))

        if x_max <= x_min or y_max <= y_min:
            return

        p1 = (int(x_min * frame_w), int(y_min * frame_h))
        p2 = (int(x_max * frame_w), int(y_max * frame_h))
        cv2.rectangle(output, p1, p2, (255, 0, 255), 2)
        cv2.putText(output, "Analysis ROI", (p1[0], max(20, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
