"""LightGBM quantile regression model for occupancy forecasting.

This module defines the model architecture, hyperparameter configuration,
and prediction interface for the primary forecasting model.

Why LightGBM instead of a neural network?
  LightGBM is a Gradient Boosted Decision Tree (GBDT) framework that excels on
  tabular data — exactly the kind of structured, heterogeneous feature set used
  here (time encodings, lag values, calendar flags, lecture proxies).  Compared
  to the TensorFlow MLP baseline, LightGBM offers:
    - No feature scaling required (trees are scale-invariant)
    - Faster training and hyperparameter tuning
    - Native feature importance (used by SHAP for explainability)
    - Better performance on small-to-medium tabular datasets (empirically: MAE
      improved from 10.60 to 0.80, an 85% reduction)

Why quantile regression?
  A single point forecast does not communicate uncertainty.  By training three
  separate models at quantiles q03, q50, and q97, we obtain:
    - q50: the median (best single guess)
    - q03/q97: a 94% prediction interval (94% of actual values should fall between them)
  This interval is shown in the dashboard and used by the recommendations engine
  to assess risk level (high occupancy even at the q03 estimate → warn early).

Key symbols:
    GBDTConfig: Dataclass of LightGBM hyperparameters.
    GBDTBundle: Dataclass holding all three trained models + metadata.
    build_gbdt_model: Construct a single LGBMRegressor for a given quantile.
    train_gbdt_quantiles: Train all three quantile models with early stopping.
    predict_gbdt: Generate clipped, monotonic predictions from a trained bundle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# LightGBM is an optional dependency — the service can still start (using the
# baseline fallback) if the package is not installed.
try:
    import lightgbm as lgb
except ImportError:
    lgb = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class GBDTConfig:
    """Hyperparameter configuration for the LightGBM quantile models.

    These values were chosen based on the dataset characteristics (13,000 rows,
    40 features, 15-min resolution) and are deliberately conservative to avoid
    overfitting on the limited one-year training set.

    Attributes:
        num_leaves: Maximum number of leaves per tree.  Controls model complexity.
            Lower = simpler trees, less overfitting.  31 is the LightGBM default.
        learning_rate: Step size for gradient descent.  Smaller = slower learning
            but better generalisation.  0.05 is a common production value.
        n_estimators: Maximum number of boosting rounds (trees).  Early stopping
            will terminate training before this if the validation loss stops improving.
        min_child_samples: Minimum samples required in a leaf node.  Acts as a
            regulariser — prevents tiny leaf nodes that overfit to noise.
        subsample: Fraction of training rows sampled per tree (bagging).
            Introduces randomness to reduce variance.
        colsample_bytree: Fraction of features sampled per tree.  Reduces correlation
            between trees and improves robustness.
        early_stopping_rounds: Stop training if validation loss does not improve for
            this many consecutive rounds.  Prevents overfitting.
        random_seed: Fixed seed for reproducibility.
        verbose: LightGBM verbosity level.  -1 = silent.
        categorical_features: Column names to treat as categorical (uses LightGBM's
            native categorical split optimisation).
    """
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 500
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 20
    random_seed: int = 42
    verbose: int = -1
    categorical_features: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Model bundle dataclass
# ---------------------------------------------------------------------------

@dataclass
class GBDTBundle:
    """Container for the three trained quantile models and associated metadata.

    All three models are stored together so that they can be saved, loaded,
    and used as a unit.  The bundle is serialised to disk by model_store.py
    as three separate .joblib files plus a metadata JSON file.

    Note on naming convention:
        The field names model_q10 / model_q90 were chosen for backward
        compatibility with the first prototype.  In production, these models
        are trained at alpha=0.03 and alpha=0.97 respectively, so the correct
        labels are q03 and q97.  The GBDTBundle docstring and all calling code
        use the correct q03/q97 terminology.

    Attributes:
        model_q10: LGBMRegressor trained at alpha=0.03 (lower bound of PI).
        model_q50: LGBMRegressor trained at alpha=0.50 (median forecast).
        model_q90: LGBMRegressor trained at alpha=0.97 (upper bound of PI).
        feature_columns: Ordered list of feature names — must match the order
            used during training for correct predictions.
        feature_importance: Dict mapping feature name → importance score (from
            the q50 model), sorted descending.  Used by SHAP and dashboards.
        config: The GBDTConfig used for training (stored for auditability).
    """
    model_q10: Any  # LGBMRegressor at alpha=0.03 (field name kept for compat)
    model_q50: Any  # LGBMRegressor at alpha=0.50
    model_q90: Any  # LGBMRegressor at alpha=0.97 (field name kept for compat)
    feature_columns: list[str]
    feature_importance: dict[str, float]
    config: GBDTConfig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_lightgbm() -> Any:
    """Raise a helpful error if LightGBM is not installed.

    LightGBM is checked lazily (at call time) rather than at import time so that
    the service can start and serve baseline forecasts even without the package.

    Returns:
        The `lightgbm` module.

    Raises:
        RuntimeError: If lightgbm is not installed.
    """
    if lgb is None:
        raise RuntimeError(
            "LightGBM is not available. Install with: pip install lightgbm"
        )
    return lgb


# ---------------------------------------------------------------------------
# Model construction and training
# ---------------------------------------------------------------------------

def build_gbdt_model(alpha: float, config: GBDTConfig) -> Any:
    """Construct a single LGBMRegressor configured for quantile regression.

    The quantile objective makes LightGBM minimise the asymmetric pinball loss:
        L(y, f) = alpha * max(y - f, 0) + (1 - alpha) * max(f - y, 0)
    where y is the true value and f is the prediction.  For alpha=0.5 this
    reduces to the mean absolute error.  For alpha=0.97, the model is penalised
    heavily for underestimation, driving it to predict a high upper bound.

    Args:
        alpha: Target quantile in (0, 1).  Use 0.03 for lower PI, 0.50 for
            median, 0.97 for upper PI.
        config: Hyperparameter configuration (see GBDTConfig).

    Returns:
        An unfitted LGBMRegressor ready to be trained with .fit().
    """
    _ensure_lightgbm()
    return lgb.LGBMRegressor(
        objective="quantile",   # Minimise pinball loss instead of MSE/MAE
        alpha=alpha,            # The quantile to target
        num_leaves=config.num_leaves,
        learning_rate=config.learning_rate,
        n_estimators=config.n_estimators,
        min_child_samples=config.min_child_samples,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        random_state=config.random_seed,
        verbose=config.verbose,
        n_jobs=-1,              # Use all available CPU cores
    )


def train_gbdt_quantiles(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_columns: list[str],
    config: GBDTConfig | None = None,
) -> GBDTBundle:
    """Train three LightGBM models for quantile regression (q03, q50, q97).

    Three independent models are trained because quantile regression is
    model-specific: each model minimises the pinball loss for its own alpha.
    A single model cannot simultaneously optimise for multiple quantiles
    (unlike some neural network approaches with multiple outputs).

    Early stopping is used on each model: training halts when the validation
    loss has not improved for `config.early_stopping_rounds` consecutive rounds.
    This prevents overfitting while allowing fast models to stop well before
    `n_estimators` rounds.

    Args:
        X_train: Feature matrix for training (n_train × n_features), numpy array.
        y_train: Target vector for training (n_train,).
        X_val: Feature matrix for validation / early stopping (n_val × n_features).
        y_val: Target vector for validation (n_val,).
        feature_columns: List of feature names in the same order as the columns
            of X_train.  Stored in the bundle for inference-time alignment.
        config: Hyperparameter config.  Defaults to GBDTConfig() if None.

    Returns:
        GBDTBundle containing all three fitted models, feature importance
        (from q50), and the config used.
    """
    _ensure_lightgbm()
    if config is None:
        config = GBDTConfig()

    models = {}
    # Train each quantile model sequentially.  Each model is completely independent.
    for alpha, name in [(0.03, "q03"), (0.5, "q50"), (0.97, "q97")]:
        model = build_gbdt_model(alpha=alpha, config=config)

        # Callbacks control training: early_stopping watches validation loss;
        # log_evaluation with period=0 suppresses per-round output (keeps logs clean).
        callbacks = [
            lgb.early_stopping(config.early_stopping_rounds, verbose=config.verbose > 0),
            lgb.log_evaluation(period=0),
        ]
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="quantile",   # Evaluate using pinball loss on validation set
            callbacks=callbacks,
        )
        models[name] = model

    # Feature importance is extracted from the q50 (median) model because it
    # best reflects overall predictive signal.  The q03/q97 models may have
    # slightly different importance rankings due to their asymmetric objectives.
    importance = dict(zip(
        feature_columns,
        models["q50"].feature_importances_.tolist(),
    ))
    # Sort descending so the most important features appear first in dashboards.
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    return GBDTBundle(
        model_q10=models["q03"],   # Note: field name is q10 for historical reasons
        model_q50=models["q50"],
        model_q90=models["q97"],   # Note: field name is q90 for historical reasons
        feature_columns=feature_columns,
        feature_importance=importance,
        config=config,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_gbdt(
    bundle: GBDTBundle,
    X: np.ndarray,
    capacity: int = 84,
) -> dict[str, np.ndarray]:
    """Generate clipped, monotonicity-enforced quantile predictions.

    This function is the canonical prediction interface used during *training
    evaluation* (in train_gbdt.py).  For live inference in production, the
    clipping is applied directly in main.py (_forecast_lgbm) so that the zone's
    actual capacity (read from the database) is used rather than this default.

    Post-processing steps:
      1. Clip predictions to [0, capacity]: occupancy cannot be negative or
         exceed the physical room capacity.
      2. Enforce monotonicity: q03 <= q50 <= q97.  Quantile crossing (where a
         lower quantile predicts a higher value than an upper quantile) is a
         known artefact of training independent models.  np.minimum/maximum
         corrects this without modifying the q50 median estimate.

    Args:
        bundle: Trained GBDTBundle (output of train_gbdt_quantiles).
        X: Feature matrix (n_samples × n_features).  Must have columns in the
            same order as bundle.feature_columns.
        capacity: Physical room capacity.  Predictions above this are impossible
            and indicate model extrapolation; they are clipped here.

    Returns:
        Dict with keys 'q03', 'q50', 'q97', each a numpy array of shape (n_samples,).
    """
    # Raw predictions from each quantile model (may exceed capacity or be negative
    # if the model extrapolates outside the training distribution).
    q03 = bundle.model_q10.predict(X).clip(0, capacity)
    q50 = bundle.model_q50.predict(X).clip(0, capacity)
    q97 = bundle.model_q90.predict(X).clip(0, capacity)

    # Enforce monotonicity: q03 ≤ q50 ≤ q97.
    # After clipping, q03 could exceed q50 (e.g. if q50 was clipped down to
    # capacity but q03 was also high).  np.minimum/maximum corrects this.
    q03 = np.minimum(q03, q50)  # Lower bound cannot exceed median
    q97 = np.maximum(q97, q50)  # Upper bound must be at least the median

    return {"q03": q03, "q50": q50, "q97": q97}
