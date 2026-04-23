#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "services" / "forecast" / "main.py"


def _load_module(module_name: str, db_path: Path, training_mode: str):
    if db_path.exists():
        db_path.unlink()
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    os.environ["FORECAST_MODEL_BACKEND"] = "tf_mlp"
    os.environ["FORECAST_TRAINING_MODE"] = training_mode
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    module.Base.metadata.create_all(bind=module.engine)
    return module


def _assert_locked_response(response) -> None:  # noqa: ANN001
    if response.status_code != 423:
        raise AssertionError(f"expected 423, got {response.status_code}: {response.text}")
    detail = str(response.json().get("detail", ""))
    if "training disabled in locked mode" not in detail:
        raise AssertionError(f"unexpected lock detail: {detail}")


def main() -> int:
    locked_module = _load_module(
        module_name="sitcheck_forecast_main_locked",
        db_path=ROOT / ".tmp_forecast_training_lock.db",
        training_mode="locked",
    )
    with TestClient(locked_module.app) as client:
        _assert_locked_response(
            client.post(
                "/v1/train",
                json={"zone_id": "default-zone", "horizon": 60, "history_hours": 24},
            )
        )
        _assert_locked_response(
            client.post(
                "/v1/train/batch",
                json={"zone_id": "default-zone", "horizons": [60], "history_hours": 24},
            )
        )
        _assert_locked_response(
            client.post(
                "/v1/train/evaluate",
                json={"zone_id": "default-zone", "horizons": [60], "history_hours": 24},
            )
        )
        _assert_locked_response(
            client.post(
                "/v1/train/promote",
                json={"zone_id": "default-zone", "run_id": "eval-12345678", "horizons": [60], "history_hours": 24},
            )
        )

    maintenance_module = _load_module(
        module_name="sitcheck_forecast_main_maintenance",
        db_path=ROOT / ".tmp_forecast_training_maintenance.db",
        training_mode="maintenance",
    )
    with TestClient(maintenance_module.app) as client:
        responses = [
            client.post("/v1/train", json={"zone_id": "default-zone", "horizon": 60, "history_hours": 24}),
            client.post("/v1/train/batch", json={"zone_id": "default-zone", "horizons": [60], "history_hours": 24}),
            client.post("/v1/train/evaluate", json={"zone_id": "default-zone", "horizons": [60], "history_hours": 24}),
            client.post(
                "/v1/train/promote",
                json={"zone_id": "default-zone", "run_id": "eval-12345678", "horizons": [60], "history_hours": 24},
            ),
        ]
        for response in responses:
            if response.status_code == 423:
                raise AssertionError(f"maintenance mode must not return 423: {response.text}")

    print("forecast training lock tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
