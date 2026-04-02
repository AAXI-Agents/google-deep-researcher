"""API key authentication and sliding-window rate limiting."""
import os
import time
from collections import defaultdict
from threading import Lock
from typing import Annotated

from fastapi import Depends, Header, HTTPException

# Shared in-memory store: api_key -> list of request timestamps within current window
_rate_limit_store: dict[str, list[float]] = defaultdict(list)
_rate_limit_lock = Lock()


def reset_rate_limits() -> None:
    """Clear all rate-limit counters. Intended for use in tests."""
    with _rate_limit_lock:
        _rate_limit_store.clear()


def get_valid_api_keys() -> set[str]:
    """Return the set of valid API keys from the API_KEYS environment variable.

    API_KEYS should be a comma-separated list of opaque key strings, e.g.::

        API_KEYS=key-abc123,key-xyz789
    """
    keys_env = os.getenv("API_KEYS", "")
    return {k.strip() for k in keys_env.split(",") if k.strip()}


def verify_api_key(x_api_key: Annotated[str | None, Header()] = None) -> str:
    """FastAPI dependency: validate the ``X-API-Key`` request header.

    Returns the key string so downstream dependencies can use it as a
    per-client identifier.

    Raises:
        HTTPException 401 – when the header is missing or the key is unknown.
    """
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="API key required")
    valid_keys = get_valid_api_keys()
    if not valid_keys or x_api_key not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


def check_rate_limit(api_key: Annotated[str, Depends(verify_api_key)]) -> str:
    """FastAPI dependency: enforce a sliding-window rate limit per API key.

    Limits are read from environment variables at call time so they can be
    overridden in tests without module reloads:

    * ``RATE_LIMIT_REQUESTS`` – maximum requests allowed per window (default 60)
    * ``RATE_LIMIT_WINDOW``   – window length in seconds (default 60)

    Note: The rate-limit counters are stored in process memory. This is
    sufficient for a single-process deployment. For multi-process or
    multi-instance deployments, replace ``_rate_limit_store`` with a shared
    backend such as Redis or MongoDB to ensure consistent enforcement across
    all workers.

    Raises:
        HTTPException 429 – when the caller has exceeded the configured limit.
    """
    max_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))
    window = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
    now = time.time()
    cutoff = now - window

    with _rate_limit_lock:
        timestamps = _rate_limit_store[api_key]
        # Prune timestamps that have fallen outside the current window
        _rate_limit_store[api_key] = [t for t in timestamps if t > cutoff]
        if len(_rate_limit_store[api_key]) >= max_requests:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(window)},
            )
        _rate_limit_store[api_key].append(now)

    return api_key
