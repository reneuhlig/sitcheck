"""Model bundle persistence: saving and loading trained forecast models.

This module abstracts all filesystem I/O for model bundles so that training
code (train_gbdt.py, train_tf.py) and inference code (main.py) never need to
know the on-disk layout directly.

Two distinct bundle formats are supported, one per model backend:

  TF-MLP bundle (save_bundle / load_bundle):
    models/{zone_id}/h{horizon}/
      model.keras          — Keras native format (primary, robust across TF versions)
      saved_model/         — TensorFlow SavedModel format (for TF tooling, optional)
      scaler.joblib        — Fitted StandardScaler (must be applied before inference)
      metadata.json        — Training config, metrics, feature list, version tag
      residuals.json       — Empirical residual quantiles q10/q90 for prediction intervals

  GBDT bundle (load_gbdt_bundle / gbdt_model_status):
    models/{zone_id}/h{horizon}/
      gbdt_q10.joblib      — LGBMRegressor at alpha=0.03 (lower bound)
      gbdt_q50.joblib      — LGBMRegressor at alpha=0.50 (median)
      gbdt_q90.joblib      — LGBMRegressor at alpha=0.97 (upper bound)
      metadata_gbdt.json   — Training metrics, feature list, promoted flag, version
      feature_importance.json — Feature importance scores from the q50 model

Both formats share the same directory layout: model_dir/{zone_id}/h{horizon}/.
This makes it easy to compare TF and GBDT bundles for the same zone/horizon.

Key functions:
    bundle_dir: Compute the bundle directory path.
    save_bundle / load_bundle: TF-MLP bundle I/O.
    load_gbdt_bundle: GBDT bundle loading (saving is in train_gbdt.py).
    model_status / gbdt_model_status: Check if a bundle exists and read its metadata.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import joblib


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _safe_name(value: str) -> str:
    """Sanitise a string for use as a filesystem directory name.

    Replaces any character that is not alphanumeric, underscore, dot, or hyphen
    with an underscore.  This prevents directory traversal and filesystem errors
    when zone_id values contain spaces or special characters.

    Args:
        value: The raw string to sanitise (e.g. a zone_id like "floor-2 west").

    Returns:
        A filesystem-safe string (e.g. "floor-2_west"), defaulting to "default"
        if the result would be empty.
    """
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "default"


def bundle_dir(model_dir: str, zone_id: str, horizon: int) -> Path:
    """Return the bundle directory Path for a given zone and horizon.

    Both TF and GBDT bundles live in the same directory; the filename extension
    distinguishes them (.keras / .joblib / metadata.json / metadata_gbdt.json).

    Args:
        model_dir: Root directory for all model storage (e.g. "./models").
        zone_id: Zone identifier string (e.g. "default-zone", "floor-2").
        horizon: Forecast horizon in minutes (e.g. 60 for H60).

    Returns:
        Absolute-or-relative Path: model_dir/{safe_zone_id}/h{horizon}/
    """
    return Path(model_dir) / _safe_name(zone_id) / f"h{horizon}"


def bundle_paths(model_dir: str, zone_id: str, horizon: int) -> dict[str, Path]:
    """Return a dict of all expected file paths for a TF-MLP bundle.

    Used by both save_bundle and load_bundle to ensure consistent file naming.

    Args:
        model_dir: Root model directory.
        zone_id: Zone identifier.
        horizon: Forecast horizon in minutes.

    Returns:
        Dict mapping role names to Path objects:
            root, keras_model, saved_model_dir, scaler, metadata, residuals.
    """
    root = bundle_dir(model_dir, zone_id, horizon)
    return {
        "root": root,
        "keras_model": root / "model.keras",           # Primary Keras format
        "saved_model_dir": root / "saved_model",        # Optional TF SavedModel format
        "scaler": root / "scaler.joblib",               # StandardScaler for feature normalisation
        "metadata": root / "metadata.json",             # Training config and metrics
        "residuals": root / "residuals.json",           # q10/q90 residuals for PI construction
    }


# ---------------------------------------------------------------------------
# TF-MLP bundle: save and load
# ---------------------------------------------------------------------------

def save_bundle(
    model_dir: str,
    zone_id: str,
    horizon: int,
    model: Any,
    scaler: Any,
    metadata: dict[str, Any],
    residuals: dict[str, Any],
) -> dict[str, str]:
    """Save a TF-MLP model bundle to disk.

    Writes five files: the Keras model (primary), an optional SavedModel export,
    the fitted StandardScaler, training metadata, and validation residuals.

    The .keras format is the primary save format because it is more portable
    and version-stable than the older HDF5 (.h5) format.  The SavedModel export
    is attempted as a best-effort for compatibility with TF Serving or TFLite;
    failures are silently ignored because it is not required for inference.

    Args:
        model_dir: Root model directory.
        zone_id: Zone identifier.
        horizon: Forecast horizon in minutes.
        model: Trained tf.keras.Model instance.
        scaler: Fitted sklearn.preprocessing.StandardScaler instance.
        metadata: Dict of training config, metrics, and version info.
        residuals: Dict with 'q10', 'q90' residual arrays and metadata.

    Returns:
        Dict mapping role names to string paths of the written files.
    """
    from model_tf import ensure_tensorflow

    paths = bundle_paths(model_dir, zone_id, horizon)
    paths["root"].mkdir(parents=True, exist_ok=True)

    # Save in Keras native format (.keras) — most robust for reload.
    model.save(paths["keras_model"])

    # Attempt SavedModel export (used by TF Serving, TFLite tools).
    # This is not required for our inference path, so failures are swallowed.
    try:
        tf = ensure_tensorflow()
        tf.saved_model.save(model, str(paths["saved_model_dir"]))
    except Exception:
        pass

    # Persist the fitted scaler so inference code can apply the same transformation
    # that was used during training.
    joblib.dump(scaler, paths["scaler"])

    # JSON files are human-readable for debugging and audit purposes.
    paths["metadata"].write_text(json.dumps(metadata, indent=2))
    paths["residuals"].write_text(json.dumps(residuals, indent=2))

    return {key: str(path) for key, path in paths.items()}


def load_bundle(model_dir: str, zone_id: str, horizon: int) -> dict[str, Any]:
    """Load a TF-MLP model bundle from disk.

    Validates that all required files exist before loading, providing a clear
    error message if the bundle is incomplete or missing.

    Args:
        model_dir: Root model directory.
        zone_id: Zone identifier.
        horizon: Forecast horizon in minutes.

    Returns:
        Dict with keys: model (Keras model), scaler, metadata, residuals, paths.

    Raises:
        FileNotFoundError: If any required bundle file is missing.
    """
    from model_tf import ensure_tensorflow

    paths = bundle_paths(model_dir, zone_id, horizon)

    # Check all required files (skip saved_model_dir and root — both are optional).
    missing = [
        key for key, path in paths.items()
        if key not in ("saved_model_dir", "root") and not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing model bundle files: {missing}")

    tf = ensure_tensorflow()
    model = tf.keras.models.load_model(paths["keras_model"])
    scaler = joblib.load(paths["scaler"])
    metadata = json.loads(paths["metadata"].read_text())
    residuals = json.loads(paths["residuals"].read_text())

    return {
        "model": model,
        "scaler": scaler,
        "metadata": metadata,
        "residuals": residuals,
        "paths": {key: str(path) for key, path in paths.items()},
    }


def model_status(model_dir: str, zone_id: str, horizon: int) -> dict[str, Any]:
    """Check whether a TF-MLP bundle exists and return its metadata.

    Used by the forecast service's /health endpoint and by the inference path
    to decide which backend is available before committing to a prediction.

    Args:
        model_dir: Root model directory.
        zone_id: Zone identifier.
        horizon: Forecast horizon in minutes.

    Returns:
        Dict with keys: exists (bool), zone_id, horizon, paths, metadata (or None).
    """
    paths = bundle_paths(model_dir, zone_id, horizon)

    # A bundle is considered valid only if all four key files exist.
    exists = (
        paths["keras_model"].exists()
        and paths["scaler"].exists()
        and paths["metadata"].exists()
        and paths["residuals"].exists()
    )

    metadata: dict[str, Any] | None = None
    if exists:
        metadata = json.loads(paths["metadata"].read_text())

    return {
        "exists": exists,
        "zone_id": zone_id,
        "horizon": horizon,
        "paths": {key: str(path) for key, path in paths.items()},
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# GBDT bundle: load and status
# (Saving is handled in train_gbdt.py:save_gbdt_bundle to keep training
#  logic co-located with training code.)
# ---------------------------------------------------------------------------

def gbdt_bundle_paths(model_dir: str, zone_id: str, horizon: int) -> dict[str, Path]:
    """Return a dict of all expected file paths for a GBDT bundle.

    Args:
        model_dir: Root model directory.
        zone_id: Zone identifier.
        horizon: Forecast horizon in minutes.

    Returns:
        Dict mapping role names to Path objects:
            root, gbdt_q10, gbdt_q50, gbdt_q90, metadata, feature_importance.
    """
    root = bundle_dir(model_dir, zone_id, horizon)
    return {
        "root": root,
        "gbdt_q10": root / "gbdt_q10.joblib",           # alpha=0.03 model
        "gbdt_q50": root / "gbdt_q50.joblib",           # alpha=0.50 model
        "gbdt_q90": root / "gbdt_q90.joblib",           # alpha=0.97 model
        "metadata": root / "metadata_gbdt.json",         # Metrics + config
        "feature_importance": root / "feature_importance.json",  # Optional
    }


def load_gbdt_bundle(model_dir: str, zone_id: str, horizon: int) -> dict[str, Any]:
    """Load a GBDT bundle from disk.

    Loads the three LightGBM models and metadata.  Feature importance is optional
    (returns an empty dict if the file does not exist) so that bundles trained
    before the feature importance file was added remain loadable.

    Args:
        model_dir: Root model directory.
        zone_id: Zone identifier.
        horizon: Forecast horizon in minutes.

    Returns:
        Dict with keys: model_q10, model_q50, model_q90, metadata,
        feature_importance, paths.

    Raises:
        FileNotFoundError: If any of the three .joblib files or metadata_gbdt.json
            are missing.
    """
    paths = gbdt_bundle_paths(model_dir, zone_id, horizon)

    # Validate required files (feature_importance is optional).
    missing = [
        key for key, path in paths.items()
        if key not in ("root", "feature_importance") and not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing GBDT bundle files: {missing}")

    # Load all three quantile models from disk.
    model_q10 = joblib.load(paths["gbdt_q10"])
    model_q50 = joblib.load(paths["gbdt_q50"])
    model_q90 = joblib.load(paths["gbdt_q90"])
    metadata = json.loads(paths["metadata"].read_text())

    # Feature importance is optional — return empty dict if absent.
    feature_importance: dict[str, float] = {}
    if paths["feature_importance"].exists():
        feature_importance = json.loads(paths["feature_importance"].read_text())

    return {
        "model_q10": model_q10,
        "model_q50": model_q50,
        "model_q90": model_q90,
        "metadata": metadata,
        "feature_importance": feature_importance,
        "paths": {key: str(path) for key, path in paths.items()},
    }


def _exception_summary(exc: Exception) -> str:
    message = str(exc).strip()
    return f"{exc.__class__.__name__}: {message}" if message else exc.__class__.__name__


def gbdt_model_status(model_dir: str, zone_id: str, horizon: int, verify_load: bool = False) -> dict[str, Any]:
    """Check whether a GBDT bundle exists and return its metadata.

    Used by the forecast service's /health endpoint to report which models
    are available and their promotion status.

    Args:
        model_dir: Root model directory.
        zone_id: Zone identifier.
        horizon: Forecast horizon in minutes.
        verify_load: If true, actually load the LightGBM bundle so dependency
            or pickle compatibility failures are surfaced by health/status.

    Returns:
        Dict with keys: exists (bool), zone_id, horizon, backend ("lgbm"),
        paths, metadata (or None if bundle does not exist), loadable, load_error.
    """
    paths = gbdt_bundle_paths(model_dir, zone_id, horizon)

    # All three model files plus metadata must exist for the bundle to be valid.
    exists = all(
        paths[key].exists()
        for key in ["gbdt_q10", "gbdt_q50", "gbdt_q90", "metadata"]
    )

    metadata: dict[str, Any] | None = None
    if exists:
        metadata = json.loads(paths["metadata"].read_text())

    missing = [
        key
        for key in ["gbdt_q10", "gbdt_q50", "gbdt_q90", "metadata"]
        if not paths[key].exists()
    ]

    loadable: bool | None = None
    load_error: str | None = None
    if verify_load:
        if not exists:
            loadable = False
            load_error = f"missing GBDT bundle files: {missing}"
        else:
            try:
                load_gbdt_bundle(model_dir, zone_id, horizon)
                loadable = True
            except Exception as exc:
                loadable = False
                load_error = _exception_summary(exc)

    return {
        "exists": exists,
        "zone_id": zone_id,
        "horizon": horizon,
        "backend": "lgbm",
        "paths": {key: str(path) for key, path in paths.items()},
        "metadata": metadata,
        "missing": missing,
        "loadable": loadable,
        "load_error": load_error,
    }
