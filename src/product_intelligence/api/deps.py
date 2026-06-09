"""Request-scoped dependencies: API-key auth and rate limiting.

Both are config-gated so local development stays frictionless (no key, no limit)
while production can require ``X-API-Key`` and cap request rate per client. The
limiter is a dependency-free sliding-window counter - fine for a single replica
and easily swapped for Redis when horizontally scaled.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request, status

from product_intelligence.core.config import settings


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not settings.auth_enabled:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": settings.api_key_header},
        )


class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            bucket = self._hits[key]
            while bucket and bucket[0] <= now - self.window:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                retry_after = int(self.window - (now - bucket[0])) + 1
                return False, retry_after
            bucket.append(now)
            return True, 0


_limiter = SlidingWindowRateLimiter(
    settings.rate_limit_requests, settings.rate_limit_window_seconds
)


async def rate_limit(request: Request) -> None:
    if not settings.rate_limit_enabled:
        return
    client = request.headers.get(settings.api_key_header) or (
        request.client.host if request.client else "anonymous"
    )
    allowed, retry_after = _limiter.check(client)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded.",
            headers={"Retry-After": str(retry_after)},
        )
