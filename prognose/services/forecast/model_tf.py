"""Removed TF-MLP model compatibility module.

The production forecast stack is LGBM-only. These symbols remain only to make
stale imports fail explicitly instead of loading TensorFlow or serving old MLPs.
"""
from __future__ import annotations

from typing import Any


TF_MLP_REMOVED_DETAIL = "TF-MLP is removed from Sitcheck forecasting; only LGBM is supported."


def ensure_tensorflow() -> Any:
    raise RuntimeError(TF_MLP_REMOVED_DETAIL)


def is_tensorflow_available() -> tuple[bool, str]:
    return False, TF_MLP_REMOVED_DETAIL


def build_mlp_model(*_: Any, **__: Any) -> Any:
    raise RuntimeError(TF_MLP_REMOVED_DETAIL)


def build_mlp_v2(*_: Any, **__: Any) -> Any:
    raise RuntimeError(TF_MLP_REMOVED_DETAIL)
