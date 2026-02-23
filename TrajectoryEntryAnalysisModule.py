from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple


@dataclass
class EntranceZoneConfig:
    mode: str
    line_p1: Tuple[float, float]
    line_p2: Tuple[float, float]
    line_entry_direction: str
    polygon_points: List[Tuple[float, float]]
    entry_polygon_points: List[Tuple[float, float]]
    exit_polygon_points: List[Tuple[float, float]]
    min_crossing_displacement_px: float = 30.0
    min_track_points: int = 6
    min_event_cooldown_frames: int = 8

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "EntranceZoneConfig":
        line = payload.get("line", {})
        polygon = payload.get("polygon", {})
        entry_polygon = payload.get("entry_polygon", {})
        exit_polygon = payload.get("exit_polygon", {})

        legacy_polygon_points = [tuple(pt) for pt in polygon.get("points", [])]
        entry_points = [tuple(pt) for pt in entry_polygon.get("points", legacy_polygon_points)]
        exit_points = [tuple(pt) for pt in exit_polygon.get("points", [])]

        return EntranceZoneConfig(
            mode=str(payload.get("mode", "line")),
            line_p1=tuple(line.get("p1", [0.35, 0.65])),
            line_p2=tuple(line.get("p2", [0.65, 0.65])),
            line_entry_direction=str(line.get("entry_direction", "negative_to_positive")),
            polygon_points=legacy_polygon_points,
            entry_polygon_points=entry_points,
            exit_polygon_points=exit_points,
            min_crossing_displacement_px=float(payload.get("min_crossing_displacement_px", 30.0)),
            min_track_points=int(payload.get("min_track_points", 6)),
            min_event_cooldown_frames=int(payload.get("min_event_cooldown_frames", 8)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "line": {
                "p1": [self.line_p1[0], self.line_p1[1]],
                "p2": [self.line_p2[0], self.line_p2[1]],
                "entry_direction": self.line_entry_direction,
            },
            "polygon": {
                "points": [[x, y] for x, y in self.polygon_points],
            },
            "entry_polygon": {
                "points": [[x, y] for x, y in self.entry_polygon_points],
            },
            "exit_polygon": {
                "points": [[x, y] for x, y in self.exit_polygon_points],
            },
            "min_crossing_displacement_px": self.min_crossing_displacement_px,
            "min_track_points": self.min_track_points,
            "min_event_cooldown_frames": self.min_event_cooldown_frames,
        }


class TrajectoryEntryAnalysisModule:
    """Analysiert Track-Trajektorien und erkennt valide Entry-Events."""

    def __init__(self, zone_config: EntranceZoneConfig, max_history: int = 40):
        self.zone_config = zone_config
        self.max_history = max_history
        self.track_history: Dict[int, Deque[Tuple[float, float]]] = {}
        self._frame_idx = 0
        self._last_event_frame_by_track: Dict[int, int] = {}
        self._last_inside_entry_by_track: Dict[int, bool] = {}
        self._last_inside_exit_by_track: Dict[int, bool] = {}

    def set_zone_config(self, zone_config: EntranceZoneConfig):
        self.zone_config = zone_config

    def update(self, tracks: List[Dict[str, Any]], frame_shape) -> List[Dict[str, Any]]:
        self._frame_idx += 1
        frame_h, frame_w = frame_shape[:2]
        events = []

        for track in tracks:
            track_id = track.get("track_id")
            center = track.get("center")
            if track_id is None or center is None:
                continue

            history = self.track_history.setdefault(track_id, deque(maxlen=self.max_history))
            history.append(center)

            if self.zone_config.mode == "dual_polygon":
                event = self._check_entry_exit_dual_polygon(track_id, history, frame_w, frame_h)
            elif self.zone_config.mode == "polygon":
                event = self._check_entry_polygon(track_id, history, frame_w, frame_h)
            else:
                event = self._check_entry_line(track_id, history, frame_w, frame_h)

            if event is not None:
                events.append(event)

        return events

    def _check_entry_line(
        self,
        track_id: int,
        history: Deque[Tuple[float, float]],
        frame_w: int,
        frame_h: int,
    ) -> Optional[Dict[str, Any]]:
        if len(history) < self.zone_config.min_track_points:
            return None

        p1, p2 = self._line_points(frame_w, frame_h)
        first = history[0]
        last = history[-1]

        first_sign = self._signed_side(first, p1, p2)
        last_sign = self._signed_side(last, p1, p2)

        if first_sign == 0 or last_sign == 0:
            return None

        crossed = (first_sign > 0 and last_sign < 0) or (first_sign < 0 and last_sign > 0)
        if not crossed:
            return None

        displacement = abs(self._signed_distance(last, p1, p2) - self._signed_distance(first, p1, p2))
        if displacement < self.zone_config.min_crossing_displacement_px:
            return None

        if self.zone_config.line_entry_direction == "negative_to_positive":
            is_entry = first_sign < 0 and last_sign > 0
        else:
            is_entry = first_sign > 0 and last_sign < 0

        if not is_entry:
            return None

        return {
            "type": "entry",
            "track_id": track_id,
            "trajectory": list(history),
            "confidence": 1.0,
            "reason": "line_crossing_in_entry_direction",
        }

    def _check_entry_exit_dual_polygon(
        self,
        track_id: int,
        history: Deque[Tuple[float, float]],
        frame_w: int,
        frame_h: int,
    ) -> Optional[Dict[str, Any]]:
        if len(history) < max(2, self.zone_config.min_track_points):
            return None

        entry_points = self.zone_config.entry_polygon_points
        exit_points = self.zone_config.exit_polygon_points
        if len(entry_points) < 3 or len(exit_points) < 3:
            return None

        cooldown = self.zone_config.min_event_cooldown_frames
        last_event_frame = self._last_event_frame_by_track.get(track_id, -10_000)
        if self._frame_idx - last_event_frame < cooldown:
            return None

        polygon_entry = [(x * frame_w, y * frame_h) for x, y in entry_points]
        polygon_exit = [(x * frame_w, y * frame_h) for x, y in exit_points]

        prev = history[-2]
        curr = history[-1]
        displacement = ((curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2) ** 0.5
        if displacement < self.zone_config.min_crossing_displacement_px * 0.25:
            return None

        prev_entry = self._last_inside_entry_by_track.get(track_id, self._point_in_polygon(prev, polygon_entry))
        prev_exit = self._last_inside_exit_by_track.get(track_id, self._point_in_polygon(prev, polygon_exit))
        curr_entry = self._point_in_polygon(curr, polygon_entry)
        curr_exit = self._point_in_polygon(curr, polygon_exit)

        self._last_inside_entry_by_track[track_id] = curr_entry
        self._last_inside_exit_by_track[track_id] = curr_exit

        crossed_entry = (not prev_entry) and curr_entry
        crossed_exit = (not prev_exit) and curr_exit

        if crossed_entry and crossed_exit:
            return None

        if crossed_entry:
            self._last_event_frame_by_track[track_id] = self._frame_idx
            return {
                "type": "entry",
                "track_id": track_id,
                "trajectory": list(history),
                "confidence": 1.0,
                "reason": "entry_polygon_outside_to_inside",
            }

        if crossed_exit:
            self._last_event_frame_by_track[track_id] = self._frame_idx
            return {
                "type": "exit",
                "track_id": track_id,
                "trajectory": list(history),
                "confidence": 1.0,
                "reason": "exit_polygon_outside_to_inside",
            }

        return None

    def _check_entry_polygon(
        self,
        track_id: int,
        history: Deque[Tuple[float, float]],
        frame_w: int,
        frame_h: int,
    ) -> Optional[Dict[str, Any]]:
        if len(history) < self.zone_config.min_track_points:
            return None

        points_norm = self.zone_config.polygon_points
        if len(points_norm) < 3:
            return None

        polygon = [(x * frame_w, y * frame_h) for x, y in points_norm]
        first = history[0]
        last = history[-1]

        # Polygon-Logik: nur outside->inside gilt als Entry.
        first_inside = self._point_in_polygon(first, polygon)
        last_inside = self._point_in_polygon(last, polygon)
        if first_inside or not last_inside:
            return None

        # Gegen Randjitter: Mindestbewegung zwischen erstem/letztem Punkt.
        dx = last[0] - first[0]
        dy = last[1] - first[1]
        if (dx * dx + dy * dy) ** 0.5 < self.zone_config.min_crossing_displacement_px:
            return None

        return {
            "type": "entry",
            "track_id": track_id,
            "trajectory": list(history),
            "confidence": 1.0,
            "reason": "polygon_outside_to_inside",
        }

    def _line_points(self, frame_w: int, frame_h: int):
        x1, y1 = self.zone_config.line_p1
        x2, y2 = self.zone_config.line_p2
        return (x1 * frame_w, y1 * frame_h), (x2 * frame_w, y2 * frame_h)

    @staticmethod
    def _signed_side(point, line_p1, line_p2) -> int:
        distance = TrajectoryEntryAnalysisModule._signed_distance(point, line_p1, line_p2)
        if distance > 0:
            return 1
        if distance < 0:
            return -1
        return 0

    @staticmethod
    def _signed_distance(point, line_p1, line_p2) -> float:
        px, py = point
        x1, y1 = line_p1
        x2, y2 = line_p2
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    @staticmethod
    def _point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        # Ray-casting: robuste Punkt-in-Polygon-Prüfung ohne zusätzliche Lib.
        x, y = point
        inside = False
        n = len(polygon)
        px1, py1 = polygon[0]
        for idx in range(1, n + 1):
            px2, py2 = polygon[idx % n]
            if y > min(py1, py2) and y <= max(py1, py2):
                if x <= max(px1, px2):
                    if py1 != py2:
                        xints = (y - py1) * (px2 - px1) / (py2 - py1) + px1
                    else:
                        xints = px1
                    if px1 == px2 or x <= xints:
                        inside = not inside
            px1, py1 = px2, py2
        return inside
