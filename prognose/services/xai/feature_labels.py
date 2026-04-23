"""German feature-label translations for the Explainability layer.

Maps technical machine-learning feature names (as used in training data and
SHAP values) to human-readable German descriptions for display in the
dashboard and in XAI explanation responses.

Design rationale:
  All feature names in the model are in English and follow a naming convention
  that is meaningful to developers but opaque to end-users ("occupancy_lag_4",
  "lecture_net_pull", "dow_sin").  This module centralises translations so that
  any component (dashboard, XAI API, LLM prompts) can display consistent,
  user-friendly labels in German without hardcoding translations in multiple places.

Key symbols:
    FEATURE_LABELS_DE: Canonical mapping from technical name to German label.
    get_label: Lookup a label by feature name and language code.
    translate_importance: Convert a feature-importance dict to labelled records.
"""
from __future__ import annotations

# Canonical mapping: technical feature name → German display label.
# Organised by feature category to match the feature set documentation
# (see features_excel.py FEATURE_COLUMNS and CLAUDE.md Feature-Engineering table).
FEATURE_LABELS_DE: dict[str, str] = {
    # Excel-native Faktoren
    "f_month": "Monatlicher Saisoneffekt",
    "f_weekday": "Wochentags-Faktor",
    "f_tod": "Tageszeit-Faktor",
    "f_weather": "Wetter-Faktor",
    "f_bridge": "Brückentag-Faktor",
    "efficiency": "Effizienz-Faktor",
    "capacity_effective": "Effektive Kapazität",
    # Binäre Features
    "bridge_day": "Brückentag",
    "winter_break": "Winterferien",
    "weather_rainy": "Regenwetter",
    "weather_sunny": "Sonniges Wetter",
    "is_partial_closure": "Teilschließung",
    # Utilization
    "utilization_pct": "Aktuelle Auslastung (%)",
    # Zeitliche Features
    "minute_sin": "Tageszeit (zyklisch, sin)",
    "minute_cos": "Tageszeit (zyklisch, cos)",
    "dow_sin": "Wochentag (zyklisch, sin)",
    "dow_cos": "Wochentag (zyklisch, cos)",
    "day_of_year_sin": "Jahreszeit (zyklisch, sin)",
    "day_of_year_cos": "Jahreszeit (zyklisch, cos)",
    "hour_of_day": "Stunde des Tages",
    "is_weekday": "Werktag (Mo-Fr)",
    # Occupancy Lags
    "occupancy_lag_1": "Belegung vor 15 Min",
    "occupancy_lag_2": "Belegung vor 30 Min",
    "occupancy_lag_3": "Belegung vor 45 Min",
    "occupancy_lag_4": "Belegung vor 1 Std",
    "occupancy_lag_8": "Belegung vor 2 Std",
    "occupancy_lag_16": "Belegung vor 4 Std",
    "occupancy_lag_day": "Belegung gestern (gleiche Uhrzeit)",
    "occupancy_lag_week": "Belegung letzte Woche (gleiche Uhrzeit)",
    # Rolling Stats
    "occupancy_roll_mean_4": "Durchschnitt letzte Stunde",
    "occupancy_roll_mean_8": "Durchschnitt letzte 2 Std",
    "occupancy_roll_mean_16": "Durchschnitt letzte 4 Std",
    "occupancy_roll_std_4": "Schwankung letzte Stunde",
    "occupancy_roll_std_8": "Schwankung letzte 2 Std",
    # Diffs
    "occupancy_diff_1": "Trend (15-Min-Änderung)",
    "occupancy_diff_4": "Trend (1-Std-Änderung)",
    # Lecture Proxies
    "lecture_density_proxy": "Vorlesungsdichte",
    "lecture_starts_proxy": "Vorlesungsstarts (nächste 60 Min)",
    "lecture_ends_proxy": "Vorlesungsenden (nächste 60 Min)",
    "lecture_heavy_proxy": "Schwere Vorlesungen",
    # Legacy features (for backward compatibility)
    "occupancy": "Aktuelle Belegung",
    "occupancy_lag_5": "Belegung vor 5 Min",
    "occupancy_lag_10": "Belegung vor 10 Min",
    "occupancy_lag_15": "Belegung vor 15 Min",
    "occupancy_lag_30": "Belegung vor 30 Min",
    "occupancy_lag_60": "Belegung vor 1 Std",
    "occupancy_roll_mean_5": "Durchschnitt letzte 5 Min",
    "occupancy_roll_mean_15": "Durchschnitt letzte 15 Min",
    "occupancy_roll_mean_60": "Durchschnitt letzte Stunde",
    "occupancy_roll_std_5": "Schwankung letzte 5 Min",
    "occupancy_roll_std_15": "Schwankung letzte 15 Min",
    "occupancy_roll_std_60": "Schwankung letzte Stunde",
    "occupancy_diff_5": "Trend (5-Min-Änderung)",
    "occupancy_diff_15": "Trend (15-Min-Änderung)",
    "quality_score": "Datenqualität",
    "quality_flag_count": "Qualitäts-Flags",
    "event_active": "Veranstaltung aktiv",
    "event_impact_sum": "Veranstaltungs-Impact",
    "lecture_count_now": "Aktive Vorlesungen",
    "lecture_starts_next_60m": "Vorlesungsstarts (nächste 60 Min)",
    "lecture_ends_next_60m": "Vorlesungsenden (nächste 60 Min)",
    "lecture_quality_score": "Vorlesungsdaten-Qualität",
    "lecture_heavy_now": "Schwere Vorlesungen (jetzt)",
    "lecture_heavy_post_60m": "Schwere Vorlesungen (letzte 60 Min beendet)",
    "lecture_pull_regular": "Vorlesungs-Sogeffekt",
    "lecture_bib_bonus": "Bibliotheks-Bonus",
    "lecture_net_pull": "Netto-Sogeffekt",
    "lecture_count_roll_60": "Vorlesungsdichte (60 Min Ø)",
    "lecture_low_period_flag": "Vorlesungsarme Phase",
}


def get_label(feature_name: str, lang: str = "de") -> str:
    """Get human-readable label for a feature.

    Args:
        feature_name: Technical feature name.
        lang: Language code ('de' supported, falls back to feature_name).

    Returns:
        Human-readable label.
    """
    if lang == "de":
        return FEATURE_LABELS_DE.get(feature_name, feature_name)
    return feature_name


def translate_importance(
    importance: dict[str, float],
    top_n: int = 10,
    lang: str = "de",
) -> list[dict[str, str | float]]:
    """Translate feature importance to human-readable format.

    Args:
        importance: Feature name -> importance value.
        top_n: Number of top features to return.
        lang: Language for labels.

    Returns:
        List of dicts with 'feature', 'label', 'importance'.
    """
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [
        {
            "feature": name,
            "label": get_label(name, lang),
            "importance": value,
        }
        for name, value in sorted_items
    ]
