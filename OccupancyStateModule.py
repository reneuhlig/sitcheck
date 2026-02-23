from datetime import datetime
from typing import Dict, Optional


class OccupancyStateModule:
    """Verwaltet den Zustand aktuell innen befindlicher Personen (Track IDs)."""

    def __init__(self, db=None):
        self.db = db
        self.last_event_time: Optional[datetime] = None
        self.current_occupancy = 0
        self.last_event_type_by_track: Dict[int, str] = {}
        self.entries_total = 0
        self.exits_total = 0

    @property
    def occupancy(self) -> int:
        return self.current_occupancy

    def initialize_from_db(self):
        if not self.db:
            return

        latest = self.db.get_latest_room_state()
        if latest:
            self.current_occupancy = int(latest["total_persons"])
            self.db.insert_room_state(
                total_persons=self.current_occupancy,
                change_reason="tracking_session_start",
                confidence=1.0,
                notes="Neustart Session; occupancy aus letztem Zustand übernommen",
            )
        else:
            self.current_occupancy = 0
            self.db.insert_room_state(
                total_persons=0,
                change_reason="initialization",
                confidence=1.0,
                notes="Initialer Zustand",
            )

    def handle_event(self, event: Dict) -> bool:
        event_type = str(event.get("type", "entry")).lower()
        if event_type not in {"entry", "exit"}:
            return False

        track_id = event.get("track_id")
        if track_id is None:
            return False

        last_type = self.last_event_type_by_track.get(track_id)
        if last_type == event_type:
            return False

        if self.db:
            tracking_event_id = self.db.insert_tracking_event(
                event_type=event_type,
                track_id=track_id,
                confidence=event.get("confidence", 1.0),
                event_data={
                    "reason": event.get("reason"),
                    "trajectory": event.get("trajectory", []),
                },
            )
            if tracking_event_id is None:
                # Bereits gezählt (Unique Constraint) oder DB-Problem.
                return False

        if event_type == "entry":
            self.current_occupancy += 1
            self.entries_total += 1
        else:
            self.current_occupancy = max(0, self.current_occupancy - 1)
            self.exits_total += 1

        self.last_event_type_by_track[track_id] = event_type
        self.last_event_time = datetime.now()

        if self.db:
            movement_id = self.db.insert_movement(
                movement_type=event_type,
                person_count=1,
                confidence=event.get("confidence", 1.0),
                detection_sequence={
                    "track_id": track_id,
                    "reason": event.get("reason"),
                    "trajectory": event.get("trajectory", []),
                },
                notes=f"YOLO track {event_type} transition",
            )

            self.db.insert_room_state(
                total_persons=self.occupancy,
                change_reason=event_type,
                movement_id=movement_id,
                confidence=event.get("confidence", 1.0),
                notes=f"track_id={track_id}",
            )

        return True
