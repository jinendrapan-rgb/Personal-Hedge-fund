"""Shared HTTP helpers: a simple token-bucket rate limiter and a retrying
GET wrapper. Kept dependency-light so tests can monkeypatch ``requests``.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger("greymatter.data")


class RateLimiter:
    """Thread-safe token bucket. ``acquire()`` blocks until a slot is free."""

    def __init__(self, rate_per_sec: float) -> None:
        self._min_interval = 1.0 / rate_per_sec if rate_per_sec > 0 else 0.0
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_allowed = now + self._min_interval


class HttpError(RuntimeError):
    """Raised on a non-retryable HTTP failure (4xx other than 429)."""


@retry(
    retry=retry_if_exception_type((requests.RequestException,)),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def get_json(
    url: str,
    *,
    limiter: RateLimiter,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Rate-limited GET returning parsed JSON, with exponential backoff.

    Retries on network errors and HTTP 429/5xx. Raises ``HttpError`` on
    other 4xx (e.g. 404) so callers can handle "not found" deliberately.
    """
    limiter.acquire()
    resp = requests.get(url, headers=headers, params=params, timeout=timeout)
    if resp.status_code == 429 or resp.status_code >= 500:
        # Surface as a retryable RequestException.
        resp.raise_for_status()
    if resp.status_code in (401, 403):
        raise HttpError(
            f"Auth/plan error {resp.status_code} for {url.split('?')[0]}: "
            f"{resp.text[:160]}. The key authenticates but the plan likely "
            f"lacks the required history depth — the spec needs Polygon "
            f"Starter (paid) or above; free/Basic tiers will not work."
        )
    if resp.status_code >= 400:
        raise HttpError(f"GET {url} -> {resp.status_code}: {resp.text[:200]}")
    return resp.json()
