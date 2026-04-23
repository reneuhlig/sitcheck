"""Removed TF-MLP trainer compatibility module.

The forecast stack is LGBM-only. This module remains only so stale imports fail
with a clear error instead of recreating old TF-MLP artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


TF_MLP_REMOVED_DETAIL = "TF-MLP training is removed; use the LGBM/GBDT pipeline."


@dataclass
class TrainingConfig:
    model_dir: str
    zone_id: str
    horizon: int
    product: str = "short_term"
    min_train_points: int = 2000
    use_calendar_features: bool = True
    include_lecture_impact: bool = True
    min_quality_score: float = 0.0
    epochs: int = 50
    batch_size: int = 128
    verbose: int = 0
    random_seed: int = 42


def train_zone_model(*_: Any, **__: Any) -> dict[str, Any]:
    raise RuntimeError(TF_MLP_REMOVED_DETAIL)
