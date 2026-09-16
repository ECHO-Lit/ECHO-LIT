"""Load Testing -- the control plane under average, peak and sustained-peak workloads.

Test Plan Section 3.1.5.  See tests/plans/3.1.5-load-testing.md.

SRS PE-1: "...shall respond within 500 ms on average under normal load."
SRS PE-3: "A 24-hour expiry on transient objects shall bound cumulative storage
growth."

The RUP template asks for workloads that represent average and peak demand and
for the system to be observed under each.  `tests/_load.py` defines the three
shapes and the driver.  Each case below runs one shape once and asserts every
criterion against that single run -- a separate case per criterion would re-run
a multi-second workload for each assertion, and caching a run across cases would
make the module order-dependent.
"""

from __future__ import annotations

import pytest

from app.core import redis as redis_module
from app.core.settings import settings
from app.core.storage import get_storage
from app.services import custom_dataset_service
from tests._load import AVERAGE, PEAK, SUSTAINED, run_profile

pytestmark = [pytest.mark.performance, pytest.mark.slow]

BUDGET_SECONDS = 0.5  # SRS PE-1
LAG_LIMIT = 0.10


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(settings, "STORAGE_LOCAL_ROOT", str(tmp_path / "objects"))
    get_storage.cache_clear()
    yield
    get_storage.cache_clear()


@pytest.fixture(autouse=True)
def isolated_datasets(tmp_path, monkeypatch):
    monkeypatch.setattr(custom_dataset_service, "SESSIONS_BASE_DIR", tmp_path / "sessions")


def _assert_budget(report, endpoints=None, *, p95_limit: float | None = None) -> None:
    for endpoint in endpoints or report.endpoints:
        stats = report.stats(endpoint)
        assert stats.mean < BUDGET_SECONDS, f"PE-1 {endpoint}: {stats.ms()}\n{report.summary()}"
    if p95_limit is not None:
        for endpoint in report.endpoints:
            stats = report.stats(endpoint)
            assert stats.p95 < p95_limit, f"p95 {endpoint}: {stats.ms()}\n{report.summary()}"


class TestAverageLoad:
    async def test_average_load_meets_the_response_time_budget(self, monkeypatch):
        """LT-01: PE-1 as written -- "on average under normal load".

        (a) no request fails; (b) every endpoint class -- session, upload,
        submit, poll, result -- averages under 500 ms; (c) the event loop never
        stalls, so no user waits on another's request.
        """
        report = await run_profile(AVERAGE, monkeypatch)
        print(report.summary())

        assert not report.errors, report.errors[:5]
        assert report.completed_jobs >= AVERAGE.users, report.summary()
        _assert_budget(report)
        assert report.lag.max < LAG_LIMIT, report.summary()


class TestPeakLoad:
    async def test_peak_load_degrades_without_failing(self, monkeypatch):
        """LT-02: five times the users, no think time, 10x the real polling rate.

        PE-1 bounds only *normal* load, so peak is held to a stricter-than-
        required reading: (a) zero errors; (b) the 500 ms mean still holds for
        the calls a running job repeats -- submit, poll, result; (c) every
        endpoint's p95 stays under one second, including the upload burst of all
        25 users arriving in the same instant, which is a start-up artifact of
        the profile rather than a steady state.
        """
        report = await run_profile(PEAK, monkeypatch)
        print(report.summary())

        assert not report.errors, report.errors[:5]
        assert report.completed_jobs >= PEAK.users, report.summary()
        _assert_budget(report, ["submit", "poll", "result"], p95_limit=1.0)


class TestSustainedPeak:
    async def test_sustained_peak_is_stable_and_bounded(self, monkeypatch):
        """LT-03: the peak workload held three times as long.

        A defect that only accumulates -- a growing index read on every poll, a
        leak per request -- is invisible in a short burst and shows here as
        drift.  (a) zero errors; (b) polling latency in the last third of the
        run is within 1.5x of the first third; (c) process RSS grows by less
        than 64 MiB; (d) PE-3 -- every key the run left in Redis carries an
        expiry, so the store cannot grow without bound.
        """
        report = await run_profile(SUSTAINED, monkeypatch)
        print(report.summary())

        assert not report.errors, report.errors[:5]
        third = SUSTAINED.duration / 3
        early = report.stats("poll", end=third)
        late = report.stats("poll", start=2 * third, end=SUSTAINED.duration)
        assert late.p50 <= 1.5 * early.p50 + 0.005, f"early {early.ms()} | late {late.ms()}"

        growth_mib = (report.rss_end - report.rss_start) / 2**20
        assert growth_mib < 64, f"RSS grew {growth_mib:.1f} MiB\n{report.summary()}"

        immortal = []
        for client in (redis_module.redis, redis_module.job_redis):
            async for key in client.scan_iter(count=1000):
                if await client.ttl(key) < 0:
                    immortal.append(key)
        assert not immortal, f"keys with no expiry: {immortal[:10]}"
