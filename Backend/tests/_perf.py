"""Shared instruments for the Section 3.1.4 / 3.1.5 performance and load modules.

Test Plan Sections 3.1.4 and 3.1.5.  See tests/plans/3.1.4-performance-profiling.md
and tests/plans/3.1.5-load-testing.md.

A plain helper module rather than a conftest, for the reason `_fixtures.py`
gives: conftest fixtures apply to the whole suite, and nothing here should change
the environment of the ~560 tests that predate these sections.

Three instruments, each chosen so an assertion measures the product and not the
test runner:

* `loop_lag_probe` -- the oracle for "does not block the interface" (PE-2,
  US-3).  The in-process ASGI app shares one event loop with the test, exactly
  as a uvicorn worker shares one loop with every connected user.  A ticker on
  that loop records how late each wake-up was; a route that runs blocking work
  on the loop shows up as one long lag, however fast the machine is.
* `sample` -- repeated timing with mean/p50/p95.  Budgets are asserted on the
  mean (the SRS PE-1 wording is "on average") and never on a single sample.
* `CountingRedis` -- counts commands per client, so an N+1 access pattern is
  asserted as a round-trip count rather than inferred from wall-clock time on
  fakeredis, where a round trip costs microseconds instead of a network hop.
"""

from __future__ import annotations

import asyncio
import gc
import statistics
import time
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


# --------------------------------------------------------------------------
# Event-loop responsiveness
# --------------------------------------------------------------------------


@dataclass
class LagProbe:
    samples: list[float] = field(default_factory=list)

    @property
    def max(self) -> float:
        return max(self.samples, default=0.0)


@asynccontextmanager
async def loop_lag_probe(interval: float = 0.005):
    """Measure the worst scheduling delay on the running loop while the body runs.

    Windows' default timer resolution is ~15.6 ms, so an idle probe already reads
    up to ~16 ms of "lag".  Thresholds in the modules sit far above that floor and
    far below the blocking cost they are meant to catch.
    """
    probe = LagProbe()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    async def tick() -> None:
        while not stop.is_set():
            started = loop.time()
            await asyncio.sleep(interval)
            probe.samples.append(loop.time() - started - interval)

    # A full-generation GC pass over the heap the *rest of the suite* built up
    # stalls the loop for ~200 ms at unpredictable moments -- observed once in
    # a full run and never in isolation.  That pause belongs to the test
    # process, not the route under test, so everything allocated before the
    # probe starts is collected and frozen out of later GC passes.
    gc.collect()
    gc.freeze()
    task = asyncio.create_task(tick())
    # Let the ticker take its first sample before the body starts, so a body
    # that blocks immediately is still observed.
    await asyncio.sleep(0)
    try:
        yield probe
    finally:
        stop.set()
        await task
        gc.unfreeze()


def blocking_cost(seconds: float, result: Any = None) -> Callable[..., Any]:
    """A synchronous stand-in with a fixed, known cost.

    Used in place of ffprobe, a broker publish or a cold dataset scan.  The
    oracle is whether that cost lands on the event loop, so the stand-in's
    duration must be deterministic rather than whatever the host happens to do.
    """

    def _cost(*args, **kwargs):
        time.sleep(seconds)
        return result(*args, **kwargs) if callable(result) else result

    return _cost


# --------------------------------------------------------------------------
# Timing statistics
# --------------------------------------------------------------------------


@dataclass
class Stats:
    samples: list[float]

    @property
    def mean(self) -> float:
        return statistics.fmean(self.samples)

    @property
    def p50(self) -> float:
        return statistics.median(self.samples)

    @property
    def p95(self) -> float:
        ordered = sorted(self.samples)
        return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]

    @property
    def max(self) -> float:
        return max(self.samples)

    def ms(self) -> str:
        return (
            f"n={len(self.samples)} mean={self.mean * 1000:.1f}ms "
            f"p50={self.p50 * 1000:.1f}ms p95={self.p95 * 1000:.1f}ms max={self.max * 1000:.1f}ms"
        )


async def sample(n: int, action: Callable[[], Awaitable[Any]], *, warmup: int = 1) -> Stats:
    """Time `action` n times after `warmup` untimed calls (import and cache warm-up)."""
    for _ in range(warmup):
        await action()
    samples = []
    for _ in range(n):
        started = time.perf_counter()
        await action()
        samples.append(time.perf_counter() - started)
    return Stats(samples)


# --------------------------------------------------------------------------
# Redis round-trip accounting
# --------------------------------------------------------------------------


class CountingRedis:
    """Delegating proxy that counts commands issued through one client.

    A pipeline is one round trip regardless of how many commands it queues, so
    `pipeline()` is counted once and the commands inside it are not.
    """

    def __init__(self, inner):
        self._inner = inner
        self.calls: Counter[str] = Counter()

    @property
    def round_trips(self) -> int:
        return sum(self.calls.values())

    def reset(self) -> None:
        self.calls.clear()

    def __getattr__(self, name):
        attribute = getattr(self._inner, name)
        if not callable(attribute):
            return attribute

        def _counted(*args, **kwargs):
            self.calls[name] += 1
            return attribute(*args, **kwargs)

        return _counted
