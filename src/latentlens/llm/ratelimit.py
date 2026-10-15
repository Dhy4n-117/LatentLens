"""Client-side rate limiting so we never trip the Gemini free-tier quota."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable


class RateLimiter:
    """Sliding-window limiter: at most ``max_calls`` per ``period`` seconds.

    A sliding window (rather than a fixed per-minute bucket) matches how the
    Gemini API counts requests, so bursts at a minute boundary can't produce a
    429. Thread-safe; the lock is never held while sleeping.

    ``clock`` and ``sleep`` are injectable so tests run instantly.
    """

    def __init__(
        self,
        max_calls: int,
        period: float = 60.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_calls < 1 or period <= 0:
            raise ValueError("max_calls must be >= 1 and period > 0")
        self.max_calls = max_calls
        self.period = period
        self._clock = clock
        self._sleep = sleep
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    @classmethod
    def per_minute(cls, rpm: int) -> RateLimiter:
        return cls(max_calls=rpm, period=60.0)

    def acquire(self) -> float:
        """Block until a call is allowed. Returns the seconds spent waiting."""
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                self._evict(now)
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return waited
                delay = self._calls[0] + self.period - now
            self._sleep(delay)
            waited += delay

    @property
    def available(self) -> int:
        """Calls that could be made right now without waiting."""
        with self._lock:
            self._evict(self._clock())
            return self.max_calls - len(self._calls)

    def _evict(self, now: float) -> None:
        while self._calls and self._calls[0] <= now - self.period:
            self._calls.popleft()
