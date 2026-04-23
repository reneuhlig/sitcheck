from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class PreparedData:
    """Container for prepared forecasting data."""

    frame: pd.DataFrame
    target_col: str
    feature_cols: list[str]
    inferred_freq: str | None
    seasonal_period: int | None
    is_subdaily: bool


def parse_feature_list(features_arg: str | None) -> list[str] | None:
    """Parse comma-separated feature CLI argument."""
    if not features_arg:
        return None
    items = [item.strip() for item in features_arg.split(",")]
    items = [item for item in items if item]
    return items or None


def _validate_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    cols = set(frame.columns)
    missing = [col for col in required if col not in cols]
    if missing:
        available = ", ".join(map(str, frame.columns.tolist()))
        raise ValueError(
            "Missing required columns: "
            f"{missing}. Available columns: [{available}]"
        )


def _median_step(index: pd.DatetimeIndex) -> pd.Timedelta | None:
    if len(index) < 3:
        return None
    deltas = index.to_series().diff().dropna()
    if deltas.empty:
        return None
    median_delta = deltas.median()
    if pd.isna(median_delta) or median_delta <= pd.Timedelta(0):
        return None
    return pd.Timedelta(median_delta)


def _infer_freq_label(index: pd.DatetimeIndex, explicit_freq: str | None) -> str | None:
    if explicit_freq:
        return explicit_freq
    inferred = pd.infer_freq(index)
    if inferred:
        return inferred
    step = _median_step(index)
    if step is None:
        return None
    total_seconds = step.total_seconds()
    if total_seconds >= 86400 and total_seconds % 86400 == 0:
        days = int(total_seconds // 86400)
        return "D" if days == 1 else f"{days}D"
    if total_seconds >= 3600 and total_seconds % 3600 == 0:
        hours = int(total_seconds // 3600)
        return "H" if hours == 1 else f"{hours}H"
    if total_seconds >= 60 and total_seconds % 60 == 0:
        minutes = int(total_seconds // 60)
        return f"{minutes}min"
    return None


def _is_subdaily(index: pd.DatetimeIndex) -> bool:
    step = _median_step(index)
    if step is None:
        return False
    return step < pd.Timedelta(days=1)


def _infer_seasonal_period(index: pd.DatetimeIndex, subdaily: bool) -> int | None:
    step = _median_step(index)
    if step is None:
        return None

    sec = step.total_seconds()
    if sec <= 0:
        return None

    if subdaily:
        periods_per_day = 86400.0 / sec
        rounded = int(round(periods_per_day))
        if rounded >= 2 and abs(periods_per_day - rounded) < 1e-6:
            return rounded
        return None

    if abs(sec - 86400.0) <= 300.0:
        return 7

    return None


def load_and_prepare_data(
    file_path: str,
    sheet: str | int = 0,
    date_col: str = "timestamp",
    target_col: str = "Auslastung",
    freq: str | None = None,
    features: list[str] | None = None,
    seasonal_period_override: int | None = None,
) -> PreparedData:
    """Load and prepare Excel time series data for forecasting."""
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        hint = "/home/kiadmin/project_sitcheck/KI_Projekt_Daten_einJahr.xlsx"
        raise FileNotFoundError(
            f"Excel file not found: '{path}'. "
            f"If you expected a default dataset, check: {hint}"
        )

    frame = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    if frame.empty:
        raise ValueError(f"Excel sheet '{sheet}' in '{path}' is empty.")

    _validate_columns(frame, [date_col, target_col])

    frame = frame.copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce", utc=True)
    frame = frame.dropna(subset=[date_col])
    if frame.empty:
        raise ValueError(
            f"No valid datetime values in '{date_col}'. "
            f"Available columns: {frame.columns.tolist()}"
        )

    frame = frame.sort_values(date_col)
    frame = frame.drop_duplicates(subset=[date_col], keep="last")
    frame = frame.set_index(date_col)

    frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")

    if features is not None:
        _validate_columns(frame.reset_index(), features)
        feature_cols = list(features)
    else:
        numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != target_col]

    for col in feature_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    work = frame[[target_col] + feature_cols].copy() if feature_cols else frame[[target_col]].copy()

    if freq:
        work = work.resample(freq).mean(numeric_only=True)

    work[target_col] = work[target_col].interpolate(method="time").ffill().bfill()
    if feature_cols:
        work[feature_cols] = work[feature_cols].interpolate(method="time").ffill().bfill()

    inferred_freq = _infer_freq_label(work.index, explicit_freq=freq)
    subdaily = _is_subdaily(work.index)

    work["dayofweek"] = work.index.dayofweek.astype(float)
    work["dayofweek_sin"] = np.sin(2.0 * np.pi * work["dayofweek"] / 7.0)
    work["dayofweek_cos"] = np.cos(2.0 * np.pi * work["dayofweek"] / 7.0)

    if subdaily:
        work["hour"] = work.index.hour.astype(float)
        work["hour_sin"] = np.sin(2.0 * np.pi * work["hour"] / 24.0)
        work["hour_cos"] = np.cos(2.0 * np.pi * work["hour"] / 24.0)

    lag_candidates = [1, 2]
    if subdaily and len(work) > 24:
        lag_candidates.append(24)
    for lag in lag_candidates:
        work[f"{target_col}_lag_{lag}"] = work[target_col].shift(lag)

    work = work.dropna()
    if work.empty:
        raise ValueError("Prepared dataframe is empty after cleaning/feature engineering.")

    final_feature_cols = [col for col in work.columns if col != target_col]
    if not final_feature_cols:
        raise ValueError(
            "No usable feature columns were produced. "
            "Provide --features or add numeric columns to the input file."
        )

    seasonal_period = (
        seasonal_period_override
        if seasonal_period_override is not None
        else _infer_seasonal_period(work.index, subdaily=subdaily)
    )

    return PreparedData(
        frame=work,
        target_col=target_col,
        feature_cols=final_feature_cols,
        inferred_freq=inferred_freq,
        seasonal_period=seasonal_period,
        is_subdaily=subdaily,
    )
