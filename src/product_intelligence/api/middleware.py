"""ASGI middleware: request-id propagation, access logging and metrics.

Every request gets a correlation id (honouring an inbound header if present),
which is bound to the logging context and echoed back on the response. Latency
and status are recorded to Prometheus and to a single structured access-log line.
"""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from product_intelligence.core.config import settings
from product_intelligence.core.logging import get_logger, request_id_var
from product_intelligence.core.metrics import REQUEST_COUNT, REQUEST_LATENCY

logger = get_logger("product_intelligence.access")


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        header = settings.request_id_header
        request_id = request.headers.get(header) or uuid.uuid4().hex
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration = time.perf_counter() - start
            route = request.scope.get("route")
            path = getattr(route, "path", None) or request.url.path
            REQUEST_COUNT.labels(request.method, path, str(status_code)).inc()
            REQUEST_LATENCY.labels(request.method, path).observe(duration)
            logger.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": status_code,
                    "duration_ms": round(duration * 1000, 2),
                },
            )
            # Attach id to the response if one was produced.
            if "response" in locals():
                response.headers[header] = request_id
            request_id_var.reset(token)
