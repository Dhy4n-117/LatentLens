from __future__ import annotations

import pytest

from latentlens.llm import RateLimiter


class FakeClock:
    """Deterministic time source: sleeping simply advances the clock."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def _limiter(clock: FakeClock, max_calls: int = 3, period: float = 60.0) -> RateLimiter:
    return RateLimiter(max_calls, period, clock=clock, sleep=clock.sleep)


def test_calls_within_budget_do_not_wait(clock: FakeClock) -> None:
    limiter = _limiter(clock)
    assert [limiter.acquire() for _ in range(3)] == [0.0, 0.0, 0.0]
    assert clock.sleeps == []
    assert limiter.available == 0


def test_blocks_until_oldest_call_leaves_window(clock: FakeClock) -> None:
    limiter = _limiter(clock)
    for t in (0.0, 10.0, 20.0):
        clock.now = t
        limiter.acquire()

    clock.now = 30.0
    waited = limiter.acquire()

    assert waited == pytest.approx(30.0)  # oldest call (t=0) expires at t=60
    assert clock.now == pytest.approx(60.0)


def test_window_slides_rather_than_resetting(clock: FakeClock) -> None:
    limiter = _limiter(clock, max_calls=2, period=10.0)
    limiter.acquire()  # t=0
    clock.now = 9.0
    limiter.acquire()  # t=9

    clock.now = 10.0
    assert limiter.available == 1  # t=0 expired; t=9 still counts
    assert limiter.acquire() == 0.0
    assert limiter.acquire() == pytest.approx(9.0)  # wait for t=9 to expire at t=19


def test_sustained_throughput_matches_budget(clock: FakeClock) -> None:
    limiter = _limiter(clock, max_calls=10, period=60.0)
    for _ in range(30):
        limiter.acquire()
    # 30 calls at 10/min: the first 10 are free, then two more full windows.
    assert clock.now == pytest.approx(120.0)


@pytest.mark.parametrize(("max_calls", "period"), [(0, 60.0), (5, 0.0)])
def test_rejects_invalid_configuration(max_calls: int, period: float) -> None:
    with pytest.raises(ValueError):
        RateLimiter(max_calls, period)


def test_per_minute_factory() -> None:
    limiter = RateLimiter.per_minute(15)
    assert (limiter.max_calls, limiter.period) == (15, 60.0)
