import time
from pathlib import Path
from typing import Dict, Any
from DataLoader import LiveImageLoader
from DetectionProcessor import DetectionProcessor


class LiveDetectionProcessor:
    def __init__(self,
                 detector,
                 db_config: Dict[str, str],
                 input_x: str,
                 input_y: str,
                 run_config: Dict[str, Any] = None):
        self.detector = detector
        self.db_config = db_config
        self.input_x = Path(input_x)
        self.input_y = Path(input_y)
        self.run_config = run_config or {}
        self.loader = LiveImageLoader(input_x, input_y)
        self.processor = DetectionProcessor(
            detector=detector,
            db_config=db_config,
            data_dir="",     # Nicht nötig, LiveLoader liefert Bilder direkt
            run_config=run_config
        )

    def start(self):
        """Startet die Live-Überwachung und sofortige Verarbeitung"""
        print(f"👀 Warte auf neue Bilder in {self.input_x} und {self.input_y} ...")
        run_id = None
        try:
            for source, img in self.loader.watch():
                if img is None:
                    continue

                # Temporäres Bild speichern (z. B. im Arbeitsspeicher oder tmp-Datei)
                temp_path = Path(f"./_live_tmp/{int(time.time() * 1000)}_{source}.jpg")
                temp_path.parent.mkdir(exist_ok=True)
                import cv2
                cv2.imwrite(str(temp_path), img)

                # DetectionProcessor einmal für dieses einzelne Bild ausführen
                if not run_id:
                    # Erzeuge einmal eine Run-ID beim Start
                    run_id = self.processor._setup_database(run_id=str(time.time()))

                print(f"▶️ Verarbeitung: {temp_path.name} ({source})")
                result = self.processor._process_single_image(
                    run_id=run_id,
                    current=1,
                    total=1,
                    image_path=str(temp_path),
                    classification=source   # Merkt sich Herkunft (input_x oder input_y)
                )

                self.processor._save_result(result, db_connected=True)

                temp_path.unlink(missing_ok=True)

        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Beendet den Live-Betrieb"""
        self.loader.stop()
        print("🛑 Live-Verarbeitung gestoppt.")
