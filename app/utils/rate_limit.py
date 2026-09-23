"""
Simple in-memory sliding-window rate limiter, keyed by client IP.
Good enough for a portfolio deployment; swap for Redis-backed limiting
if this ever needed to survive multiple server processes.
"""
import time
from collections import defaultdict, deque
from fastapi import Request, HTTPException

from app.config import RATE_LIMIT_PER_MINUTE

_request_log = defaultdict(deque)


def enforce_rate_limit(request: Request):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = _request_log[client_ip]

    # drop anything older than 60 seconds
    while window and window[0] < now - 60:
        window.popleft()

    if len(window) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {RATE_LIMIT_PER_MINUTE} requests/minute",
        )

    window.append(now)
