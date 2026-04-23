# Lecture Impact Contract (Forecast Features)

This document defines the minute-level contract from `services/lecture-ingest` to forecast features.

## Scope

- Source: `https://api.dhbw.app/rapla/MA/lectures` (via ingest service, no HTML scraping).
- Storage target: `lecture_activity.metadata` (no schema change required).
- Usage: exogenous forecast features only (no synthetic writes into `counts`).

## Defaults

- `LECTURE_EFFECT_ENABLED=true`
- `LECTURE_AVG_ATTENDANCE=20`
- `LECTURE_HEAVY_BIB_PERSONS=4`
- `LECTURE_HEAVY_WINDOW_MINUTES=60`
- `LECTURE_ONSITE_TYPES=PRESENCE`
- `LECTURE_HEAVY_PHRASES=logik und algebra,operations research,grundlagen und logik,maschinelles lernen,machine learning,deep learning,digitale signalverarbeitung`
- `LECTURE_HEAVY_KEYWORDS=mathe,mathematik,statistik,algorithm,theorie,theoret,physik,analysis,lineare algebra,regelungstechnik`
- `LECTURE_IMPACT_MODEL_VERSION=lecture-impact-v1`

## Heavy classification order

1. Normalize title (`NFKD -> ASCII -> lowercase -> whitespace collapse`).
2. Check `LECTURE_HEAVY_PHRASES` via phrase contains.
3. If no phrase hit, check `LECTURE_HEAVY_KEYWORDS` via contains.

Note:

- No generic standalone `logik` keyword is used to avoid false positives such as `Handelslogik / eCommerce`.

## Minute-level Definitions

For each minute `t`:

- `L_t`: active onsite lectures.
- `H_t`: active heavy lectures.
- `H_post_t`: heavy lectures ended in the last 60 minutes.

Derived signals:

- `lecture_pull_regular_t = 20 * L_t`
- `heavy_bib_bonus_t = 4 * (H_t + H_post_t)`
- `lecture_net_pull_t = lecture_pull_regular_t - heavy_bib_bonus_t`

Interpretation:

- Lectures reduce library demand.
- Heavy modules compensate that reduction during lecture and 60 minutes after lecture end.

## Metadata Fields in `lecture_activity.points[*].metadata`

- `heavy_active_lectures` (int)
- `heavy_ended_last_60m` (int)
- `lecture_pull_regular` (float)
- `heavy_bib_bonus` (float)
- `lecture_net_pull` (float)
- `impact_model_version` (string)

Additional config trace fields are included for debugging:

- `lecture_effect_enabled`
- `lecture_avg_attendance`
- `lecture_heavy_bib_persons`
- `lecture_heavy_window_minutes`
- `lecture_heavy_phrases`
- `lecture_onsite_types`
- `lecture_heavy_keywords`

## Forecast Feature Mapping

`services/forecast/features.py` maps metadata into:

- `lecture_heavy_now`
- `lecture_heavy_post_60m`
- `lecture_pull_regular`
- `lecture_bib_bonus`
- `lecture_net_pull`

Fallback behavior:

- Missing metadata does not crash inference/training.
- Heavy signals fallback to `0`.
- `lecture_pull_regular` falls back to `active_lectures * 20`.
