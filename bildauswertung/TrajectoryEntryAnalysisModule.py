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
        self._recent_lost_tracks: Dict[int, Dict[str, Any]] = {}
        self.reid_max_gap_frames = 12
        self.reid_max_distance_px = 70.0
        self._frame_idx = 0
        self._last_event_frame_by_track: Dict[int, int] = {}
        self._last_inside_entry_by_track: Dict[int, bool] = {}
        self._last_inside_exit_by_track: Dict[int, bool] = {}
        self._last_seen_primary_zone_by_track: Dict[int, str] = {}
        self._zone_label_history_by_track: Dict[int, Deque[str]] = {}
        self._global_zone_history: Deque[str] = deque(maxlen=6)
        self._last_global_primary_zone: str = "none"
        self._last_global_event_frame: int = -10_000

    def set_zone_config(self, zone_config: EntranceZoneConfig):
        self.zone_config = zone_config

    def update(self, tracks: List[Dict[str, Any]], frame_shape) -> List[Dict[str, Any]]:
        self._frame_idx += 1
        frame_h, frame_w = frame_shape[:2]
        events = []
        active_track_ids = set()

        self.reid_max_distance_px = max(
            40.0,
            self.zone_config.min_crossing_displacement_px * 2.0,
        )

        for track in tracks:
            if bool(track.get("is_stale", False)):
                continue
            track_id = track.get("track_id")
            center = track.get("center")
            if track_id is None or center is None:
                continue

            active_track_ids.add(int(track_id))
            self._try_stitch_track_identity(int(track_id), center)

            history = self.track_history.setdefault(track_id, deque(maxlen=self.max_history))
            history.append(center)
            motion_direction = str(track.get("motion_direction", "still"))
            motion_magnitude = float(track.get("motion_magnitude", 0.0))

            if self.zone_config.mode == "dual_polygon":
                event = self._check_entry_exit_dual_polygon(
                    track_id,
                    history,
                    frame_w,
                    frame_h,
                    motion_direction,
                    motion_magnitude,
                )
            elif self.zone_config.mode == "polygon":
                event = self._check_entry_polygon(track_id, history, frame_w, frame_h)
            else:
                event = self._check_entry_line(track_id, history, frame_w, frame_h)

            if event is not None:
                events.append(event)

        if self.zone_config.mode == "dual_polygon" and not events:
            global_event = self._check_global_dual_polygon_transition(active_track_ids)
            if global_event is not None:
                events.append(global_event)

        newly_lost_track_ids = self._update_recent_lost_tracks(active_track_ids)
        if self.zone_config.mode == "dual_polygon" and not events and newly_lost_track_ids:
            lost_track_event = self._check_lost_track_zone_event(newly_lost_track_ids, frame_h)
            if lost_track_event is not None:
                events.append(lost_track_event)

        return events

    def _check_entry_line(
        self,
        track_id: int,
        history: Deque[Tuple[float, float]],
        frame_w: int,
        frame_h: int,
    ) -> Optional[Dict[str, Any]]:
        if len(history) < max(3, self.zone_config.min_track_points // 2):
            return None

        cooldown = self.zone_config.min_event_cooldown_frames
        last_event_frame = self._last_event_frame_by_track.get(track_id, -10_000)
        if self._frame_idx - last_event_frame < cooldown:
            return None

        p1, p2 = self._line_points(frame_w, frame_h)
        prev = history[-2]
        curr = history[-1]
        prev_sign = self._signed_side(prev, p1, p2)
        curr_sign = self._signed_side(curr, p1, p2)

        if prev_sign == 0 or curr_sign == 0:
            return None

        crossed = (prev_sign > 0 and curr_sign < 0) or (prev_sign < 0 and curr_sign > 0)
        if not crossed:
            return None

        lookback = min(5, len(history) - 1)
        anchor = history[-(lookback + 1)]
        displacement = abs(self._signed_distance(curr, p1, p2) - self._signed_distance(anchor, p1, p2))
        if displacement < self.zone_config.min_crossing_displacement_px * 0.50:
            return None

        move_vec = (curr[0] - anchor[0], curr[1] - anchor[1])
        line_vec = (p2[0] - p1[0], p2[1] - p1[1])
        line_normal = (-line_vec[1], line_vec[0])
        normal_projection = (move_vec[0] * line_normal[0]) + (move_vec[1] * line_normal[1])
        if abs(normal_projection) < 1e-3:
            return None

        if self.zone_config.line_entry_direction == "negative_to_positive":
            if prev_sign < 0 and curr_sign > 0:
                event_type = "entry"
                reason = "line_crossing_negative_to_positive"
            else:
                event_type = "exit"
                reason = "line_crossing_positive_to_negative"
        else:
            if prev_sign > 0 and curr_sign < 0:
                event_type = "entry"
                reason = "line_crossing_positive_to_negative"
            else:
                event_type = "exit"
                reason = "line_crossing_negative_to_positive"

        self._last_event_frame_by_track[track_id] = self._frame_idx

        return {
            "type": event_type,
            "track_id": track_id,
            "trajectory": list(history),
            "confidence": 1.0,
            "reason": reason,
        }

    def _check_entry_exit_dual_polygon(
        self,
        track_id: int,
        history: Deque[Tuple[float, float]],
        frame_w: int,
        frame_h: int,
        motion_direction: str,
        motion_magnitude: float,
    ) -> Optional[Dict[str, Any]]:
        if len(history) < max(3, self.zone_config.min_track_points // 2):
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

        curr = history[-1]
        tolerance_px = max(8.0, self.zone_config.min_crossing_displacement_px * 0.35)
        current_zone = self._classify_dual_zone(curr, polygon_entry, polygon_exit, tolerance_px)

        if current_zone == "none":
            return None

        zone_history = self._zone_label_history_by_track.setdefault(track_id, deque(maxlen=6))
        zone_history.append(current_zone)
        stable_zone = self._dominant_zone(zone_history)
        if stable_zone == "none":
            return None

        last_seen_zone = self._last_seen_primary_zone_by_track.get(track_id)
        self._last_seen_primary_zone_by_track[track_id] = stable_zone

        if last_seen_zone is None or last_seen_zone == stable_zone:
            return None

        if {last_seen_zone, stable_zone} != {"entry", "exit"}:
            return None

        lookback = min(5, len(history) - 1)
        start_point = history[-(lookback + 1)]
        move_vector = (curr[0] - start_point[0], curr[1] - start_point[1])
        move_length = (move_vector[0] ** 2 + move_vector[1] ** 2) ** 0.5
        if move_length < self.zone_config.min_crossing_displacement_px * 0.15:
            return None

        if last_seen_zone == "exit" and stable_zone == "entry":
            event_type = "entry"
            reason = "dual_polygon_transition_exit_to_entry"
        elif last_seen_zone == "entry" and stable_zone == "exit":
            event_type = "exit"
            reason = "dual_polygon_transition_entry_to_exit"
        else:
            return None

        self._last_event_frame_by_track[track_id] = self._frame_idx
        return {
            "type": event_type,
            "track_id": track_id,
            "trajectory": list(history),
            "confidence": 1.0,
            "reason": reason,
        }

    def _check_global_dual_polygon_transition(self, active_track_ids: set[int]) -> Optional[Dict[str, Any]]:
        if not active_track_ids:
            return None
        if len(active_track_ids) > 3:
            return None

        active_zones = []
        for track_id in active_track_ids:
            zone = self._last_seen_primary_zone_by_track.get(track_id, "none")
            if zone in {"entry", "exit"}:
                active_zones.append(zone)

        if not active_zones:
            return None

        entry_votes = sum(1 for z in active_zones if z == "entry")
        exit_votes = sum(1 for z in active_zones if z == "exit")
        current_global_zone = "entry" if entry_votes >= exit_votes else "exit"
        self._global_zone_history.append(current_global_zone)
        stable_global_zone = self._dominant_zone(self._global_zone_history)
        if stable_global_zone not in {"entry", "exit"}:
            return None

        previous_global_zone = self._last_global_primary_zone
        self._last_global_primary_zone = stable_global_zone
        if previous_global_zone == "none" or previous_global_zone == stable_global_zone:
            return None

        cooldown = max(self.zone_config.min_event_cooldown_frames, 18)
        if (self._frame_idx - self._last_global_event_frame) < cooldown:
            return None

        if previous_global_zone == "exit" and stable_global_zone == "entry":
            event_type = "entry"
            reason = "dual_polygon_global_transition_exit_to_entry"
        elif previous_global_zone == "entry" and stable_global_zone == "exit":
            event_type = "exit"
            reason = "dual_polygon_global_transition_entry_to_exit"
        else:
            return None

        representative_track_id = min(active_track_ids)
        self._last_global_event_frame = self._frame_idx
        return {
            "type": event_type,
            "track_id": representative_track_id,
            "trajectory": [],
            "confidence": 0.75,
            "reason": reason,
        }

    def _try_stitch_track_identity(self, track_id: int, center: Tuple[float, float]):
        if track_id in self.track_history:
            return
        if not self._recent_lost_tracks:
            return

        best_old_id = None
        best_distance = float("inf")

        for old_id, payload in list(self._recent_lost_tracks.items()):
            gap = self._frame_idx - int(payload.get("frame_idx", -10_000))
            if gap < 0 or gap > self.reid_max_gap_frames:
                continue
            old_center = payload.get("center")
            if old_center is None:
                continue
            distance = ((float(center[0]) - float(old_center[0])) ** 2 + (float(center[1]) - float(old_center[1])) ** 2) ** 0.5
            if distance < best_distance and distance <= self.reid_max_distance_px:
                best_distance = distance
                best_old_id = old_id

        if best_old_id is None:
            return

        previous_history = self.track_history.pop(best_old_id, None)
        if previous_history is not None:
            self.track_history[track_id] = deque(previous_history, maxlen=self.max_history)

        if best_old_id in self._last_event_frame_by_track:
            self._last_event_frame_by_track[track_id] = self._last_event_frame_by_track.pop(best_old_id)
        if best_old_id in self._last_seen_primary_zone_by_track:
            self._last_seen_primary_zone_by_track[track_id] = self._last_seen_primary_zone_by_track.pop(best_old_id)
        if best_old_id in self._zone_label_history_by_track:
            self._zone_label_history_by_track[track_id] = self._zone_label_history_by_track.pop(best_old_id)
        if best_old_id in self._last_inside_entry_by_track:
            self._last_inside_entry_by_track[track_id] = self._last_inside_entry_by_track.pop(best_old_id)
        if best_old_id in self._last_inside_exit_by_track:
            self._last_inside_exit_by_track[track_id] = self._last_inside_exit_by_track.pop(best_old_id)

        self._recent_lost_tracks.pop(best_old_id, None)

    def _check_lost_track_zone_event(self, newly_lost_track_ids: List[int], frame_h: int) -> Optional[Dict[str, Any]]:
        for track_id in newly_lost_track_ids:
            history = self.track_history.get(track_id)
            if not history or len(history) < max(8, self.zone_config.min_track_points + 2):
                continue

            last_event_frame = self._last_event_frame_by_track.get(track_id, -10_000)
            if (self._frame_idx - last_event_frame) < self.zone_config.min_event_cooldown_frames:
                continue

            dominant_zone = self._last_seen_primary_zone_by_track.get(track_id, "none")
            if dominant_zone not in {"entry", "exit"}:
                continue

            lookback = min(5, len(history) - 1)
            anchor = history[-(lookback + 1)]
            curr = history[-1]
            dy = float(curr[1]) - float(anchor[1])
            abs_dy = abs(dy)
            min_vertical_motion = self.zone_config.min_crossing_displacement_px * 0.05
            normalized_y = float(curr[1]) / max(1.0, float(frame_h))

            if dominant_zone == "exit":
                if dy < -min_vertical_motion and normalized_y < 0.48:
                    self._last_event_frame_by_track[track_id] = self._frame_idx
                    return {
                        "type": "exit",
                        "track_id": track_id,
                        "trajectory": list(history),
                        "confidence": 0.65 if abs_dy < min_vertical_motion else 0.75,
                        "reason": "dual_polygon_lost_track_in_exit_zone",
                    }
            else:
                if dy > min_vertical_motion and normalized_y > 0.54:
                    self._last_event_frame_by_track[track_id] = self._frame_idx
                    return {
                        "type": "entry",
                        "track_id": track_id,
                        "trajectory": list(history),
                        "confidence": 0.65 if abs_dy < min_vertical_motion else 0.75,
                        "reason": "dual_polygon_lost_track_in_entry_zone",
                    }

        return None

    def _update_recent_lost_tracks(self, active_track_ids: set[int]) -> List[int]:
        newly_lost_track_ids: List[int] = []
        known_ids = list(self.track_history.keys())
        for known_id in known_ids:
            if known_id in active_track_ids:
                self._recent_lost_tracks.pop(known_id, None)
                continue
            history = self.track_history.get(known_id)
            if not history:
                continue
            if known_id not in self._recent_lost_tracks:
                newly_lost_track_ids.append(known_id)
            self._recent_lost_tracks[known_id] = {
                "center": history[-1],
                "frame_idx": self._frame_idx,
            }

        stale_ids = []
        for lost_id, payload in self._recent_lost_tracks.items():
            gap = self._frame_idx - int(payload.get("frame_idx", -10_000))
            if gap > self.reid_max_gap_frames:
                stale_ids.append(lost_id)
        for stale_id in stale_ids:
            self._recent_lost_tracks.pop(stale_id, None)

        return newly_lost_track_ids

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

    def _classify_dual_zone(
        self,
        point: Tuple[float, float],
        polygon_entry: List[Tuple[float, float]],
        polygon_exit: List[Tuple[float, float]],
        tolerance_px: float,
    ) -> str:
        inside_entry = self._point_in_polygon(point, polygon_entry)
        inside_exit = self._point_in_polygon(point, polygon_exit)

        if inside_entry and inside_exit:
            entry_center = self._polygon_centroid(polygon_entry)
            exit_center = self._polygon_centroid(polygon_exit)
            dist_entry = ((point[0] - entry_center[0]) ** 2 + (point[1] - entry_center[1]) ** 2) ** 0.5
            dist_exit = ((point[0] - exit_center[0]) ** 2 + (point[1] - exit_center[1]) ** 2) ** 0.5
            return "entry" if dist_entry <= dist_exit else "exit"
        if inside_entry:
            return "entry"
        if inside_exit:
            return "exit"

        dist_entry = self._distance_to_polygon_boundary(point, polygon_entry)
        dist_exit = self._distance_to_polygon_boundary(point, polygon_exit)
        if min(dist_entry, dist_exit) <= tolerance_px:
            return "entry" if dist_entry <= dist_exit else "exit"

        return "none"

    @staticmethod
    def _dominant_zone(zone_history: Deque[str]) -> str:
        if not zone_history:
            return "none"
        entry_votes = sum(1 for z in zone_history if z == "entry")
        exit_votes = sum(1 for z in zone_history if z == "exit")
        if entry_votes >= 2 and entry_votes > exit_votes:
            return "entry"
        if exit_votes >= 2 and exit_votes > entry_votes:
            return "exit"
        return zone_history[-1]

    @staticmethod
    def _distance_to_polygon_boundary(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> float:
        if len(polygon) < 2:
            return 10_000.0
        min_dist = 10_000.0
        for idx in range(len(polygon)):
            p1 = polygon[idx]
            p2 = polygon[(idx + 1) % len(polygon)]
            dist = TrajectoryEntryAnalysisModule._point_to_segment_distance(point, p1, p2)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    @staticmethod
    def _point_to_segment_distance(point, seg_a, seg_b) -> float:
        px, py = point
        ax, ay = seg_a
        bx, by = seg_b
        abx = bx - ax
        aby = by - ay
        apx = px - ax
        apy = py - ay
        ab_len_sq = (abx * abx) + (aby * aby)
        if ab_len_sq <= 1e-6:
            return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
        t = ((apx * abx) + (apy * aby)) / ab_len_sq
        t = max(0.0, min(1.0, t))
        cx = ax + (t * abx)
        cy = ay + (t * aby)
        return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5

    @staticmethod
    def _polygon_centroid(polygon: List[Tuple[float, float]]) -> Tuple[float, float]:
        if not polygon:
            return (0.0, 0.0)
        sx = sum(p[0] for p in polygon)
        sy = sum(p[1] for p in polygon)
        n = float(len(polygon))
        return (sx / n, sy / n)
