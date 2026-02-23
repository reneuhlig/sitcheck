import time
from typing import Dict, Optional, Any, Callable

from DatabaseHandler import DatabaseHandler
from VideoInputModule import VideoInputModule
from YOLOTrackingModule import YOLOTrackingModule
from TrajectoryEntryAnalysisModule import (
    EntranceZoneConfig,
    TrajectoryEntryAnalysisModule,
)
from OccupancyStateModule import OccupancyStateModule
from VisualizationOutputModule import VisualizationOutputModule


class LiveProcessor:
    """Live-Orchestrierung für YOLO-Tracking und Entry-Pass-by-Analyse."""

    def __init__(
        self,
        detector,
        video_source: str,
        zone_config: EntranceZoneConfig,
        tracker_config: str = "bytetrack.yaml",
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        image_size: int = 640,
        tta_enabled: bool = False,
        max_detections: int = 300,
        stabilization_enabled: bool = True,
        track_hold_frames: int = 5,
        box_ema_alpha: float = 0.65,
        hold_confidence_decay: float = 0.85,
        trail_length: int = 12,
        motion_min_pixels: float = 2.0,
        process_every_n_frames: int = 1,
        preprocess_enabled: bool = False,
        preprocess_upscale: float = 1.0,
        preprocess_clahe_clip: float = 2.0,
        preprocess_denoise: bool = False,
        reconnect_delay: float = 1.0,
        max_retries: int = 0,
        show_window: bool = True,
        window_name: str = "Library Entry Tracking",
        enable_zone_editor: bool = True,
        on_zone_changed: Optional[Callable[[EntranceZoneConfig], None]] = None,
        db_config: Optional[Dict[str, Any]] = None,
    ):
        self.video_input = VideoInputModule(
            source=video_source,
            reconnect_delay=reconnect_delay,
            max_retries=max_retries,
        )

        self.tracking_module = YOLOTrackingModule(
            detector=detector,
            tracker_config=tracker_config,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            image_size=image_size,
            tta_enabled=tta_enabled,
            max_detections=max_detections,
            stabilization_enabled=stabilization_enabled,
            track_hold_frames=track_hold_frames,
            box_ema_alpha=box_ema_alpha,
            hold_confidence_decay=hold_confidence_decay,
            trail_length=trail_length,
            motion_min_pixels=motion_min_pixels,
            preprocess_enabled=preprocess_enabled,
            preprocess_upscale=preprocess_upscale,
            preprocess_clahe_clip=preprocess_clahe_clip,
            preprocess_denoise=preprocess_denoise,
        )
        self.process_every_n_frames = max(1, int(process_every_n_frames))
        self._last_tracks = []

        self.entry_analysis = TrajectoryEntryAnalysisModule(zone_config=zone_config)
        self.zone_config = zone_config

        self.db = None
        if db_config:
            self.db = DatabaseHandler(**db_config)

        self.occupancy_state = OccupancyStateModule(db=self.db)
        self.visualization = VisualizationOutputModule(
            show_window=show_window,
            window_name=window_name,
            enable_zone_editor=enable_zone_editor,
            on_zone_changed=self._handle_zone_changed,
        )
        self._on_zone_changed = on_zone_changed

        self.running = False

    def start(self):
        if self.db:
            if not self.db.connect():
                print("[ERROR] Datenbankverbindung fehlgeschlagen")
                return
            if not self.db.create_tables():
                print("[ERROR] Tabellen konnten nicht erstellt werden")
                return
            self.occupancy_state.initialize_from_db()

        if not self.video_input.open():
            print("[ERROR] Videoquelle konnte nicht geöffnet werden")
            return

        print("\n" + "=" * 80)
        print("[SYSTEM] LIVE TRACKING GESTARTET (YOLO TRACK)")
        print("=" * 80)
        print(f"  Modell: {self.tracking_module.detector.model_name} v{self.tracking_module.detector.model_version}")
        print(f"  Tracker: {self.tracking_module.tracker_config}")
        print(f"  Nur Klasse: person (COCO=0)")
        print("  Beenden: Taste 'q' oder ESC")
        print("=" * 80 + "\n")

        self.running = True
        frame_idx = 0

        try:
            while self.running:
                ok, frame = self.video_input.read()
                if not ok or frame is None:
                    print("[WARN] Kein Frame verfügbar – retry...")
                    time.sleep(0.05)
                    continue

                frame_idx += 1
                run_tracking_now = (frame_idx % self.process_every_n_frames) == 0
                if run_tracking_now:
                    tracks = self.tracking_module.track(frame)
                    self._last_tracks = tracks
                else:
                    tracks = self._last_tracks

                events = self.entry_analysis.update(tracks=tracks, frame_shape=frame.shape) if run_tracking_now else []
                frame_entries = 0
                frame_exits = 0

                for event in events:
                    if self.occupancy_state.handle_event(event):
                        event_type = str(event.get("type", "entry")).upper()
                        if event_type == "ENTRY":
                            frame_entries += 1
                        elif event_type == "EXIT":
                            frame_exits += 1
                        print(
                            f"[{event_type}] Frame={frame_idx} | TrackID={event['track_id']} | "
                            f"Occupancy={self.occupancy_state.occupancy}"
                        )

                frame_h, frame_w = frame.shape[:2]
                vis_frame = self.visualization.draw(
                    frame=frame,
                    tracks=tracks,
                    zone_config=self.zone_config,
                    occupancy=self.occupancy_state.occupancy,
                    entries_total=self.occupancy_state.entries_total,
                    exits_total=self.occupancy_state.exits_total,
                    events_in_frame={"entry": frame_entries, "exit": frame_exits},
                )
                self.visualization.show(vis_frame)

                if self.visualization.show_window:
                    key = self.visualization.wait_key(1)
                    self.visualization.handle_key(key)
                    if self.visualization.should_quit(key):
                        break

                if not self.visualization.show_window:
                    time.sleep(0.001)

        except KeyboardInterrupt:
            print("\n[INFO] Abbruch durch Benutzer")
        finally:
            self.stop()

    def _handle_zone_changed(self, new_zone_config: EntranceZoneConfig):
        self.zone_config = new_zone_config
        self.entry_analysis.set_zone_config(new_zone_config)
        if self._on_zone_changed:
            self._on_zone_changed(new_zone_config)

    def stop(self):
        self.running = False
        self.video_input.release()
        self.visualization.close()
        if self.db:
            self.db.close()
        print(
            f"[INFO] Tracking beendet | Entries={self.occupancy_state.entries_total} | "
            f"Exits={self.occupancy_state.exits_total} | "
            f"Occupancy={self.occupancy_state.occupancy}"
        )
