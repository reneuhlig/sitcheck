"""SHAP-based model explainability for LightGBM GBDT forecasts.

SHAP (SHapley Additive exPlanations) provides a mathematically rigorous way to
attribute a model's prediction to individual input features.  For tree models
(LightGBM, XGBoost), TreeExplainer computes exact Shapley values in polynomial
time — much faster than the kernel-based method used for black-box models.

How SHAP values work:
  For a prediction yhat, each feature i receives a SHAP value phi_i such that:
      yhat = base_value + sum(phi_i for all features)
  where base_value is the model's mean prediction over the training set.
  A positive phi_i means the feature pushed the prediction UP relative to base;
  a negative phi_i means it pushed the prediction DOWN.

This module wraps SHAP TreeExplainer in a service-friendly API and translates
feature names to German labels using feature_labels.py.

SHAP is an optional dependency (XAI_SHAP_ENABLED=true required).  When not
available, the XAI service falls back to heuristic proxy drivers.

Key symbols:
    SHAPExplainer: Wrapper class for TreeExplainer with local/global explain APIs.
"""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    import shap
except ImportError:
    shap = None  # type: ignore[assignment]

from feature_labels import FEATURE_LABELS_DE, get_label, translate_importance


def _ensure_shap() -> Any:
    """Return the shap module or raise RuntimeError if not installed."""
    if shap is None:
        raise RuntimeError("SHAP is not available. Install with: pip install shap")
    return shap


class SHAPExplainer:
    """SHAP TreeExplainer wrapper for LightGBM quantile models.

    Wraps shap.TreeExplainer to provide a simplified API for generating both:
      - Local explanations: per-prediction feature contributions (Shapley values)
      - Global explanations: mean absolute SHAP values across a background dataset

    Designed to work with the q50 (median) LightGBM model from the GBDT bundle,
    because the q50 model represents the central prediction that users see.
    The q03/q97 models could also be explained, but the median is most intuitive.
    """

    def __init__(self, model: Any, feature_columns: list[str]):
        """Initialise the SHAP explainer with a trained LightGBM model.

        Args:
            model: Trained LightGBM model (typically the q50 model from GBDTBundle).
            feature_columns: Ordered list of feature names matching model's training columns.
        """
        _ensure_shap()
        self.model = model
        self.feature_columns = feature_columns
        # TreeExplainer uses the exact tree structure to compute SHAP values
        # without sampling — this gives exact Shapley values for tree models.
        self.explainer = shap.TreeExplainer(model)

    def explain_local(
        self,
        X: np.ndarray,
        top_n: int = 5,
        lang: str = "de",
    ) -> list[dict[str, Any]]:
        """Generate local (per-prediction) SHAP explanations.

        For each sample in X, computes which features pushed the prediction
        up or down relative to the model's average prediction (base_value).
        Returns only the top_n features ranked by absolute SHAP value.

        The returned 'direction' field ('up' or 'down') is user-friendly:
          - 'up' means the feature increased occupancy relative to baseline
          - 'down' means the feature decreased occupancy relative to baseline

        Args:
            X: Feature matrix (n_samples, n_features) — must match training feature order.
            top_n: Maximum number of top features to include per sample.
            lang: Language code for label translation ('de' = German).

        Returns:
            List of explanation dicts, one per sample.  Each dict contains:
                base_value, prediction, drivers (top_n features), total_shap_sum.
        """
        shap_values = self.explainer.shap_values(X)
        base_value = float(self.explainer.expected_value)

        explanations = []
        for i in range(len(X)):
            sv = shap_values[i]
            # Sort features by absolute SHAP value, largest first, take top_n.
            sorted_idx = np.argsort(np.abs(sv))[::-1][:top_n]

            drivers = []
            for idx in sorted_idx:
                feat_name = self.feature_columns[idx]
                drivers.append({
                    "feature": feat_name,
                    "label": get_label(feat_name, lang),
                    "shap_value": float(sv[idx]),
                    "feature_value": float(X[i, idx]),
                    "direction": "up" if sv[idx] > 0 else "down",
                })

            explanations.append({
                "base_value": base_value,
                "prediction": float(base_value + np.sum(sv)),
                "drivers": drivers,
                "total_shap_sum": float(np.sum(sv)),
            })

        return explanations

    def explain_global(
        self,
        X_background: np.ndarray,
        top_n: int = 15,
        lang: str = "de",
    ) -> dict[str, Any]:
        """Generate global feature importance using mean absolute SHAP values.

        Computes SHAP values for all samples in X_background, then averages
        the absolute values per feature.  This gives a model-level view of
        which features matter most overall (vs local, which explains one prediction).

        Mean |SHAP| is preferred over simple feature importance (split count,
        gain) because it has a consistent unit (impact on prediction scale)
        and is independent of the number of features in the model.

        Args:
            X_background: Background dataset representing typical inputs
                (e.g. recent occupancy rows).  Larger = more stable estimate.
            top_n: Maximum number of features to return (ranked by importance).
            lang: Language code for label translation.

        Returns:
            Dict with: method, n_samples, features (labelled list), raw_importance.
        """
        shap_values = self.explainer.shap_values(X_background)
        # Average absolute SHAP value per feature across all background samples.
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        importance = dict(zip(self.feature_columns, mean_abs_shap.tolist()))
        sorted_importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )

        return {
            "method": "shap_tree_explainer",
            "n_samples": len(X_background),
            "features": translate_importance(sorted_importance, top_n=top_n, lang=lang),
            "raw_importance": {k: float(v) for k, v in sorted_importance.items()},
        }

    def compute_uncertainty_from_quantiles(
        self,
        q10: float,
        q50: float,
        q90: float,
    ) -> dict[str, Any]:
        """Compute uncertainty level from the model's quantile predictions.

        Uses the prediction interval width (q90 - q10) relative to the
        point forecast (q50) to classify uncertainty as low/medium/high.

        This is better than the heuristic XAI driver formula (which uses
        interval_width from a separate API call) because it uses the model's
        own quantile outputs directly, giving a self-consistent uncertainty
        estimate without an extra HTTP round-trip.

        Thresholds (relative spread = interval / q50):
            < 0.3  → low uncertainty    (tight interval relative to prediction)
            < 0.6  → medium uncertainty
            ≥ 0.6  → high uncertainty   (wide interval, e.g. during unusual periods)

        Args:
            q10: Lower prediction interval bound (α=0.03 model output).
            q50: Point forecast / median (α=0.50 model output).
            q90: Upper prediction interval bound (α=0.97 model output).

        Returns:
            Dict with: level, confidence, spread, relative_spread, q10, q50, q90, method.
        """
        spread = q90 - q10
        # Relative spread normalises by the point prediction so that the thresholds
        # are independent of the absolute occupancy level.
        relative_spread = spread / max(q50, 1.0)

        if relative_spread < 0.3:
            level = "low"
            confidence = "high"
        elif relative_spread < 0.6:
            level = "medium"
            confidence = "medium"
        else:
            level = "high"
            confidence = "low"

        return {
            "level": level,
            "confidence": confidence,
            "spread": float(spread),
            "relative_spread": float(relative_spread),
            "q10": float(q10),
            "q50": float(q50),
            "q90": float(q90),
            "method": "quantile_regression",
        }
