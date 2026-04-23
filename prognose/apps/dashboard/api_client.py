from __future__ import annotations

from datetime import UTC, datetime, timedelta
import os
import time
from typing import Any

import requests


DEFAULT_FORECAST_HORIZON_MINUTES = int(
    os.getenv("DEFAULT_FORECAST_HORIZON_MINUTES", os.getenv("TF_DEFAULT_HORIZON", "210"))
)


class SitcheckApiClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    def _request_timeout(self, timeout: float | None = None) -> tuple[float, float]:
        read_timeout = float(timeout if timeout is not None else self.timeout)
        # Keep connect timeout short, allow longer read timeout for heavy endpoints.
        return (5.0, max(1.0, read_timeout))

    @staticmethod
    def _effective_timeout(timeout: float | None, minimum: float) -> float:
        if timeout is None:
            return float(minimum)
        return float(timeout)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
        retries: int = 1,
    ) -> Any:
        attempts = max(1, int(retries) + 1)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = requests.request(
                    method=method,
                    url=f"{self.base_url}{path}",
                    params=params,
                    json=payload,
                    timeout=self._request_timeout(timeout),
                )
                response.raise_for_status()
                return response.json()
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as exc:
                last_error = exc
                if attempt >= attempts - 1:
                    raise
                # Short jitter-free backoff for transient local API hiccups.
                time.sleep(min(0.3, 0.1 * (attempt + 1)))
            except Exception as exc:
                last_error = exc
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"{method} {path} failed without explicit error")

    def _get(self, path: str, params: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        return self._request("GET", path, params=params, timeout=timeout)

    def _post(self, path: str, payload: dict[str, Any], timeout: float | None = None) -> Any:
        return self._request("POST", path, payload=payload, timeout=timeout)

    @staticmethod
    def _fallback_weekly_forecast(zone_id: str, days: int, slot_minutes: int, reason: str) -> dict[str, Any]:
        horizon_minutes = max(1, int(days) * 24 * 60)
        return {
            "zone_id": zone_id,
            "horizon": horizon_minutes,
            "days": int(days),
            "slot_minutes": int(slot_minutes),
            "generated_at": datetime.now(UTC).isoformat(),
            "summary": f"Weekly forecast unavailable; fallback in use ({reason}).",
            "model_version": "fallback-weekly-v1",
            "points": [],
            "evidence": {
                "evidence_id": f"weekly-fallback-{int(time.time())}",
                "generated_at": datetime.now(UTC).isoformat(),
                "time_window": {
                    "from": datetime.now(UTC).isoformat(),
                    "to": datetime.now(UTC).isoformat(),
                },
                "sources": [],
                "model": {
                    "name": "weekly_forecaster",
                    "version": "fallback-weekly-v1",
                    "backend": "fallback",
                },
                "quality": {
                    "score": 0.0,
                    "flags": ["WEEKLY_FORECAST_UNAVAILABLE"],
                },
            },
            "source": "fallback",
        }

    @staticmethod
    def _fallback_model_lineage(zone_id: str, product: str, reason: str) -> dict[str, Any]:
        return {
            "zone_id": zone_id,
            "product": product,
            "status": "fallback",
            "reason": reason,
            "model": {
                "name": "unknown",
                "version": "n/a",
                "backend": "n/a",
                "promoted": False,
            },
            "training": {
                "run_id": None,
                "trained_at": None,
                "window": None,
                "feature_set": None,
                "data_references": [],
            },
            "references": [],
        }

    @staticmethod
    def _derive_lineage_from_status(zone_id: str, product: str, status: dict[str, Any]) -> dict[str, Any]:
        metadata = status.get("metadata") if isinstance(status, dict) else {}
        metadata = metadata if isinstance(metadata, dict) else {}
        scientific_validation = metadata.get("scientific_validation") if isinstance(metadata.get("scientific_validation"), dict) else {}

        run_id = scientific_validation.get("run_id") or metadata.get("promoted_by_run_id")
        model_version = metadata.get("model_version") or f"{product}-v1"
        promoted = bool(metadata.get("promoted", False))
        backend = metadata.get("backend") or status.get("backend") or "unknown"

        training_window = {
            "history_hours": metadata.get("history_hours"),
            "horizon": metadata.get("horizon"),
            "include_lecture_impact": metadata.get("include_lecture_impact"),
        }
        if all(value is None for value in training_window.values()):
            training_window = {}

        data_references: list[dict[str, Any]] = []
        evidence = metadata.get("evidence") if isinstance(metadata.get("evidence"), dict) else {}
        if isinstance(evidence, dict):
            for source in evidence.get("sources", []) if isinstance(evidence.get("sources"), list) else []:
                if not isinstance(source, dict):
                    continue
                data_references.append(
                    {
                        "label": str(source.get("type") or "source"),
                        "source_id": str(source.get("id") or "unknown"),
                        "note": source.get("note"),
                    }
                )

        return {
            "zone_id": zone_id,
            "product": product,
            "status": "ok",
            "model": {
                "name": metadata.get("model_name") or "lgbm_quantile_forecaster",
                "version": str(model_version),
                "backend": str(backend),
                "promoted": promoted,
                "run_id": run_id,
                "test_status": metadata.get("test_status"),
            },
            "training": {
                "run_id": run_id,
                "trained_at": scientific_validation.get("promoted_at"),
                "window": training_window,
                "feature_set": metadata.get("feature_set"),
                "data_references": data_references,
            },
            "references": data_references,
            "scientific_validation": scientific_validation,
        }

    @staticmethod
    def _normalize_lineage_payload(zone_id: str, product: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return SitcheckApiClient._fallback_model_lineage(zone_id=zone_id, product=product, reason="invalid_payload")
        if "model" in payload and "training" in payload:
            return payload

        reference_objects = payload.get("reference_objects", []) if isinstance(payload.get("reference_objects"), list) else []
        training_window = {
            "history_from": payload.get("history_from"),
            "history_to": payload.get("history_to"),
            "horizon": payload.get("horizon"),
            "feature_set_version": payload.get("feature_set_version"),
            "include_lecture_impact": payload.get("include_lecture_impact"),
        }

        return {
            "zone_id": zone_id,
            "product": product,
            "status": str(payload.get("status") or "ok"),
            "model": {
                "name": payload.get("product") or product,
                "version": str(payload.get("model_version") or "n/a"),
                "backend": str(payload.get("model_backend") or "unknown"),
                "promoted": bool(payload.get("promoted", False)),
                "run_id": payload.get("model_run_id"),
                "scientific_status": payload.get("scientific_status"),
            },
            "training": {
                "run_id": payload.get("model_run_id"),
                "trained_at": payload.get("created_at") or payload.get("promoted_at"),
                "window": training_window,
                "feature_set": payload.get("feature_set_version"),
                "data_references": reference_objects,
            },
            "references": reference_objects,
            "scientific_validation": {
                "evaluation_run_id": payload.get("evaluation_run_id"),
                "scientific_status": payload.get("scientific_status"),
                "promotion_source": payload.get("promotion_source"),
            },
            "raw": payload,
        }

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def get_zones(self) -> list[dict[str, Any]]:
        return self._get("/api/v1/zones")

    def tool_get_history(
        self,
        zone_id: str,
        minutes: int = 180,
        granularity: str = "1m",
        timeout: float | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(UTC)
        start = now - timedelta(minutes=minutes)
        return self._get(
            "/api/v1/counts",
            params={
                "zone_id": zone_id,
                "from": start.isoformat(),
                "to": now.isoformat(),
                "granularity": granularity,
            },
            timeout=timeout,
        )

    def tool_get_forecast(
        self,
        zone_id: str,
        horizon: int = DEFAULT_FORECAST_HORIZON_MINUTES,
    ) -> dict[str, Any]:
        return self._get("/api/v1/forecast", params={"zone_id": zone_id, "horizon": horizon})

    def get_forecast_latest(
        self,
        zone_id: str,
        horizon: int = DEFAULT_FORECAST_HORIZON_MINUTES,
        stale_seconds: int = 900,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return self._get(
            "/api/v1/forecast/latest",
            params={"zone_id": zone_id, "horizon": horizon, "stale_seconds": stale_seconds},
            timeout=timeout,
        )

    def get_weekly_forecast_latest(
        self,
        zone_id: str,
        days: int = 7,
        slot_minutes: int = 60,
        stale_seconds: int = 900,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        horizon_minutes = max(1, int(days) * 24 * 60)
        try:
            return self._get(
                "/api/v1/forecast/weekly/latest",
                params={
                    "zone_id": zone_id,
                    "days": int(days),
                    "slot_minutes": int(slot_minutes),
                    "stale_seconds": stale_seconds,
                },
                timeout=timeout,
            )
        except Exception as exc:
            try:
                weekly = self.get_forecast_latest(
                    zone_id=zone_id,
                    horizon=horizon_minutes,
                    stale_seconds=stale_seconds,
                    timeout=timeout,
                )
            except Exception as fallback_exc:
                return self._fallback_weekly_forecast(
                    zone_id=zone_id,
                    days=days,
                    slot_minutes=slot_minutes,
                    reason=f"{exc.__class__.__name__}/{fallback_exc.__class__.__name__}",
                )

            weekly = dict(weekly)
            weekly.setdefault("days", int(days))
            weekly.setdefault("slot_minutes", int(slot_minutes))
            weekly.setdefault("source", "horizon-fallback")
            return weekly

    def get_weekly_forecast_history(
        self,
        zone_id: str,
        from_iso: str,
        to_iso: str,
        days: int = 7,
        slot_minutes: int = 60,
        limit: int = 100,
        stale_seconds: int = 900,
    ) -> dict[str, Any]:
        horizon_minutes = max(1, int(days) * 24 * 60)
        try:
            return self._get(
                "/api/v1/forecast/weekly/history",
                params={
                    "zone_id": zone_id,
                    "days": int(days),
                    "slot_minutes": int(slot_minutes),
                    "from": from_iso,
                    "to": to_iso,
                    "limit": limit,
                    "stale_seconds": stale_seconds,
                },
            )
        except Exception:
            try:
                return self.get_forecast_history(
                    zone_id=zone_id,
                    from_iso=from_iso,
                    to_iso=to_iso,
                    horizon=horizon_minutes,
                    limit=limit,
                    stale_seconds=stale_seconds,
                )
            except Exception:
                return {
                    "zone_id": zone_id,
                    "days": int(days),
                    "slot_minutes": int(slot_minutes),
                    "from": from_iso,
                    "to": to_iso,
                    "items": [],
                }

    def get_weekly_explainability(
        self,
        zone_id: str,
        days: int = 7,
        slot_minutes: int = 60,
        stale_seconds: int = 900,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return self._get(
            "/api/v1/explain/weekly",
            params={
                "zone_id": zone_id,
                "days": int(days),
                "slot_minutes": int(slot_minutes),
                "stale_seconds": stale_seconds,
            },
            timeout=self._effective_timeout(timeout, max(self.timeout, 30.0)),
        )

    def get_model_lineage(
        self,
        zone_id: str,
        product: str = "short_term",
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            payload = self._get(
                "/api/v1/models/lineage/latest",
                params={"zone_id": zone_id, "product": product},
                timeout=timeout,
            )
            return self._normalize_lineage_payload(zone_id=zone_id, product=product, payload=payload)
        except Exception as exc:
            horizon = 10080 if product == "weekly_slot" else DEFAULT_FORECAST_HORIZON_MINUTES
            try:
                status = self._get(
                    "/api/v1/model/status",
                    params={"zone_id": zone_id, "horizon": horizon},
                    timeout=timeout,
                )
            except Exception as fallback_exc:
                return self._fallback_model_lineage(
                    zone_id=zone_id,
                    product=product,
                    reason=f"{exc.__class__.__name__}/{fallback_exc.__class__.__name__}",
                )
            return self._derive_lineage_from_status(zone_id=zone_id, product=product, status=status)

    def get_command_center(
        self,
        zone_id: str,
        horizon: int = DEFAULT_FORECAST_HORIZON_MINUTES,
        history_minutes: int = 180,
        stale_seconds: int = 900,
        long_term_days: int = 14,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return self._get(
            "/api/v1/dashboard/command-center",
            params={
                "zone_id": zone_id,
                "horizon": horizon,
                "history_minutes": history_minutes,
                "stale_seconds": stale_seconds,
                "long_term_days": long_term_days,
            },
            timeout=self._effective_timeout(timeout, max(self.timeout, 35.0)),
        )

    def get_forecast_history(
        self,
        zone_id: str,
        from_iso: str,
        to_iso: str,
        horizon: int = DEFAULT_FORECAST_HORIZON_MINUTES,
        limit: int = 100,
        stale_seconds: int = 900,
    ) -> dict[str, Any]:
        return self._get(
            "/api/v1/forecast/history",
            params={
                "zone_id": zone_id,
                "horizon": horizon,
                "from": from_iso,
                "to": to_iso,
                "limit": limit,
                "stale_seconds": stale_seconds,
            },
        )

    def tool_explain_forecast(
        self,
        zone_id: str,
        horizon: int = DEFAULT_FORECAST_HORIZON_MINUTES,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return self._get(
            "/api/v1/explain",
            params={"zone_id": zone_id, "horizon": horizon},
            timeout=self._effective_timeout(timeout, max(self.timeout, 30.0)),
        )

    def get_explain_context(
        self,
        zone_id: str,
        horizon: int = DEFAULT_FORECAST_HORIZON_MINUTES,
        audience: str = "ops",
        language: str = "de",
        query: str = "",
    ) -> dict[str, Any]:
        return self._get(
            "/api/v1/explain/context",
            params={
                "zone_id": zone_id,
                "horizon": horizon,
                "audience": audience,
                "language": language,
                "query": query,
            },
        )

    def generate_explain_narrative(
        self,
        zone_id: str,
        horizon: int = DEFAULT_FORECAST_HORIZON_MINUTES,
        audience: str = "ops",
        query: str = "",
        language: str = "de",
        response_mode: str = "free",
        ollama_model: str | None = None,
        require_ollama: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "zone_id": zone_id,
            "horizon": horizon,
            "audience": audience,
            "query": query,
            "language": language,
            "response_mode": response_mode,
            "require_ollama": bool(require_ollama),
        }
        if ollama_model:
            payload["ollama_model"] = str(ollama_model)
        return self._post(
            "/api/v1/explain/narrative",
            payload,
            timeout=max(self.timeout, 180.0),
        )

    def preview_explain_prompt(
        self,
        zone_id: str,
        horizon: int = DEFAULT_FORECAST_HORIZON_MINUTES,
        audience: str = "ops",
        query: str = "",
        language: str = "de",
    ) -> dict[str, Any]:
        return self._post(
            "/api/v1/explain/prompt/preview",
            {
                "zone_id": zone_id,
                "horizon": horizon,
                "audience": audience,
                "query": query,
                "language": language,
            },
        )

    def tool_recommend_actions(
        self,
        zone_id: str,
        horizon: int = DEFAULT_FORECAST_HORIZON_MINUTES,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return self._get(
            "/api/v1/recommendations",
            params={"zone_id": zone_id, "horizon": horizon},
            timeout=self._effective_timeout(timeout, max(self.timeout, 25.0)),
        )

    def tool_simulate_scenario(self, zone_id: str, horizon: int, changes: dict[str, Any], persist: bool = False) -> dict[str, Any]:
        return self._post(
            "/api/v1/scenarios/simulate",
            {
                "zone_id": zone_id,
                "horizon": horizon,
                "persist": persist,
                "changes": changes,
            },
        )

    def list_calendar_events(self, zone_id: str, hours: int = 24) -> list[dict[str, Any]]:
        now = datetime.now(UTC)
        end = now + timedelta(hours=hours)
        return self._get(
            "/api/v1/calendar/events",
            params={
                "zone_id": zone_id,
                "from": now.isoformat(),
                "to": end.isoformat(),
            },
        )
