#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "services" / "forecast-trainer" / "main.py"
    spec = importlib.util.spec_from_file_location("sitcheck_forecast_trainer_main", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = ""

    def json(self):
        return self._payload


def _eval_payload(run_id: str, mae: float, pinball: float, coverage: float) -> dict:
    return {
        "run_id": run_id,
        "comparison": {
            "primary_horizon": 60,
            "improvement_vs_baseline_mae": {
                "tf_mlp": 0.1,
            },
        },
        "decision": {
            "scientific_pass": True,
            "champion_model": "tf_mlp",
        },
        "models": {
            "tf_mlp": {
                "horizons": {
                    "60": {
                        "test_metrics": {
                            "mae": mae,
                            "pinball": pinball,
                            "coverage90": coverage,
                        }
                    }
                }
            }
        },
    }


def main() -> int:
    module = _load_module()

    module.FORECAST_TRAINER_HORIZONS = [60]
    module.FORECAST_TRAINER_PRIMARY_HORIZON = 60
    module.FORECAST_TRAINER_RUN_ABLATION = True

    called_flags: list[bool] = []

    def _fake_post(url, json=None, timeout=None):  # noqa: ANN001
        include_lecture = bool((json or {}).get("include_lecture_impact", True))
        called_flags.append(include_lecture)
        if include_lecture:
            return _FakeResponse(_eval_payload("run-with-lecture", mae=8.0, pinball=1.1, coverage=0.90))
        return _FakeResponse(_eval_payload("run-without-lecture", mae=10.0, pinball=1.5, coverage=0.85))

    original_post = module.requests.post
    module.requests.post = _fake_post
    try:
        result = module._run_evaluation_pair(trigger="unit-test")
    finally:
        module.requests.post = original_post

    if called_flags != [True, False]:
        raise AssertionError(f"expected evaluation order [True, False], got {called_flags}")

    if result.get("with_lecture_run_id") != "run-with-lecture":
        raise AssertionError("with_lecture_run_id mismatch")
    if result.get("without_lecture_run_id") != "run-without-lecture":
        raise AssertionError("without_lecture_run_id mismatch")

    ablation = result.get("ablation", {})
    if not isinstance(ablation, dict):
        raise AssertionError("ablation summary missing")

    mae_gain = ablation.get("mae_gain_primary_horizon")
    pinball_gain = ablation.get("pinball_gain_primary_horizon")
    coverage_delta = ablation.get("coverage_delta_primary_horizon")

    if abs(float(mae_gain) - 2.0) > 1e-9:
        raise AssertionError(f"unexpected mae_gain_primary_horizon: {mae_gain}")
    if abs(float(pinball_gain) - 0.4) > 1e-9:
        raise AssertionError(f"unexpected pinball_gain_primary_horizon: {pinball_gain}")
    if abs(float(coverage_delta) - 0.05) > 1e-9:
        raise AssertionError(f"unexpected coverage_delta_primary_horizon: {coverage_delta}")

    print("forecast trainer ablation tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
