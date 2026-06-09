"""Prometheus metric definitions for the inference service.

Centralising metric objects avoids duplicate-registration errors and gives a
single import surface for instrumentation across middleware, inference and
pipelines. When ``prometheus_client`` is unavailable the no-op shims keep the
rest of the codebase import-safe.
"""

from __future__ import annotations

try:  # pragma: no cover - exercised indirectly
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _PROM_AVAILABLE = True
except Exception:  # pragma: no cover
    _PROM_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain"

    class _Noop:
        def __init__(self, *_, **__):
            pass

        def labels(self, *_, **__):
            return self

        def inc(self, *_, **__):
            return None

        def observe(self, *_, **__):
            return None

        def set(self, *_, **__):
            return None

    Counter = Gauge = Histogram = _Noop  # type: ignore

    def generate_latest(*_args, **_kwargs):  # type: ignore
        return b""

    class CollectorRegistry:  # type: ignore
        pass


REQUEST_COUNT = Counter(
    "pulse360_http_requests_total",
    "Total HTTP requests processed.",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "pulse360_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
PREDICTIONS_TOTAL = Counter(
    "pulse360_predictions_total",
    "Total account scoring predictions, partitioned by risk band.",
    ["risk_band", "model_version"],
)
CHURN_SCORE = Histogram(
    "pulse360_churn_probability",
    "Distribution of predicted churn probabilities.",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
INFERENCE_LATENCY = Histogram(
    "pulse360_inference_duration_seconds",
    "Model scoring latency (excludes HTTP overhead).",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)
DRIFT_PSI = Gauge(
    "pulse360_feature_drift_psi",
    "Population Stability Index per monitored feature.",
    ["feature"],
)
MODEL_INFO = Gauge(
    "pulse360_model_info",
    "Loaded model metadata (value is always 1).",
    ["model", "version"],
)


def metrics_available() -> bool:
    return _PROM_AVAILABLE


def render_latest() -> bytes:
    return generate_latest()
