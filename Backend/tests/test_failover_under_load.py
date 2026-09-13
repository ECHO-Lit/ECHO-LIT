"""Failover and Recovery -- component failure while users are working.

Test Plan Section 3.1.7.  See tests/plans/3.1.7-failover-and-recovery.md.
Carried from 3.1.5 ("failure injection under load").

The other 3.1.7 modules fail one component under one request.  Here the
3.1.5 workload shapes run -- each virtual user its own session, submitting,
polling and fetching -- while a fault is injected on a schedule: Redis goes
away for a second, half the workers die holding jobs, Redis restarts.

The oracle is the users' experience across the whole run:

  * during the fault, the only failure anyone sees is a clean 503 that the
    browser can read and retry (JSON detail, Retry-After, CORS header);
  * outside the fault window, no failure at all;
  * every job the API accepted finishes successfully;
  * every user carries on working after the fault, in the same session.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.core.settings import settings
from app.core.storage import get_storage
from app.services import custom_dataset_service
from tests._faults import ORIGIN, FaultHarness, run_with_faults
from tests._load import Profile

pytestmark = [pytest.mark.failover, pytest.mark.slow]

# 3.1.5's AVERAGE and PEAK shapes, shortened: the fault, not the duration, is the subject.
AVERAGE = Profile("average+fault", users=5, duration=4.0, think=0.5, poll_interval=0.2, service_time=0.1, workers=4)
PEAK = Profile("peak+fault", users=25, duration=4.0, think=0.0, poll_interval=0.05, service_time=0.05, workers=8)
FAULT_AT = 1.5
OUTAGE = 1.0


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(settings, "STORAGE_LOCAL_ROOT", str(tmp_path / "objects"))
    monkeypatch.setattr(custom_dataset_service, "SESSIONS_BASE_DIR", tmp_path / "sessions")
    get_storage.cache_clear()
    yield
    get_storage.cache_clear()


@pytest.fixture(autouse=True)
def harness(monkeypatch):
    faults = FaultHarness()
    faults.install(monkeypatch)
    yield faults
    faults.reconnect()


def _outage(harness):
    async def schedule(report, worker, origin):
        await asyncio.sleep(FAULT_AT)
        start = time.perf_counter() - origin
        harness.disconnect()
        try:
            await asyncio.sleep(OUTAGE)
        finally:
            harness.reconnect()
            report.windows.append((start, time.perf_counter() - origin))
    return schedule


def _assert_users_recovered(report) -> None:
    assert report.errors == [], report.errors[:5]
    assert report.accepted, "no job was accepted"
    unfinished = {job: status for job, status in report.accepted.items() if status != "success"}
    assert unfinished == {}
    users_after = set(report.completed_after)
    assert users_after == set(range(report.profile.users)), f"users that resumed: {sorted(users_after)}"


def _assert_only_clean_503s_inside_the_window(report) -> None:
    assert report.statuses() <= {200, 201, 202, 503}, report.statuses()
    unavailable = report.unavailable()
    assert unavailable, "the outage was not observed by any user"
    assert report.outside_windows(unavailable) == []
    for observation in unavailable:
        assert observation.retry_after, observation
        assert observation.allow_origin == ORIGIN, observation
        assert observation.json_detail, observation


class TestRedisOutageUnderLoad:
    async def test_a_redis_outage_under_average_load(self, harness, monkeypatch):
        """FO-90: guards BUG-50 under load; carried from 3.1.5.

        A one-second Redis outage in the middle of the AVERAGE workload.  Before
        the fix every request in the window was an unreadable 500 -- which the
        virtual user, like a browser, cannot retry -- and the run ended with
        users stranded mid-job.
        """
        report, worker = await run_with_faults(AVERAGE, monkeypatch, _outage(harness))
        print(f"{AVERAGE.name}: {len(report.accepted)} jobs, {len(report.unavailable())} x 503, "
              f"worker retries={worker.retries}, elapsed={report.elapsed:.1f}s")

        _assert_only_clean_503s_inside_the_window(report)
        _assert_users_recovered(report)

    async def test_a_redis_outage_under_peak_load(self, harness, monkeypatch):
        """FO-91: guards BUG-50 at PEAK -- 25 users polling twenty times faster than a browser."""
        report, worker = await run_with_faults(PEAK, monkeypatch, _outage(harness))
        print(f"{PEAK.name}: {len(report.accepted)} jobs, {len(report.unavailable())} x 503, "
              f"worker retries={worker.retries}, elapsed={report.elapsed:.1f}s")

        _assert_only_clean_503s_inside_the_window(report)
        _assert_users_recovered(report)


class TestWorkerLossUnderLoad:
    # Asserts PE-1's wall-clock budget, so it belongs to the untraced timing run
    # (TEST-05, tests/plans/4-deliverables.md).
    @pytest.mark.performance
    async def test_half_the_workers_die_holding_jobs(self, harness, monkeypatch):
        """FO-92: RE-1 under load -- "Queued work shall continue on remaining workers".

        Half the simulated workers are killed while each holds a job; their
        messages go back on the queue (`reject_on_worker_lost`) and the rest
        finish everything.  The API itself never notices: no failure is seen,
        and the per-job calls stay within PE-1's 500 ms.
        """
        async def kill_half(report, worker, origin):
            await asyncio.sleep(FAULT_AT)
            target = worker.concurrency // 2
            killed = 0
            deadline = time.perf_counter() + 1.0
            while killed < target and time.perf_counter() < deadline:
                killed += worker.kill(target - killed)
                await asyncio.sleep(0.005)
            assert killed == target, f"only {killed} of {target} workers were caught holding a job"

        report, worker = await run_with_faults(PEAK, monkeypatch, kill_half)
        durations = {name: [o.ended - o.started for o in report.observations if o.endpoint == name]
                     for name in ("submit", "poll")}
        print(f"{PEAK.name} kill: {len(report.accepted)} jobs, redelivered={worker.redelivered}, "
              + ", ".join(f"{n} mean={sum(d) / len(d) * 1000:.0f}ms" for n, d in durations.items()))

        assert worker.redelivered == PEAK.workers // 2
        assert report.unavailable() == []
        _assert_users_recovered(report)
        for name, samples in durations.items():
            assert sum(samples) / len(samples) < 0.5, name


class TestRestartUnderLoad:
    async def test_a_redis_restart_with_replay_under_load(self, harness, monkeypatch):
        """FO-93: RE-2 under load -- Redis restarts from its append-only file mid-run.

        Connections to the old process die, so requests in flight at that
        instant may see a 503; nothing else does.  Every user resumes in the
        same session with the audio it uploaded before the restart, and every
        accepted job finishes.
        """
        async def restart(report, worker, origin):
            await asyncio.sleep(FAULT_AT)
            at = time.perf_counter() - origin
            harness.restart(persist=True)
            report.windows.append((at, at))

        report, worker = await run_with_faults(AVERAGE, monkeypatch, restart)
        print(f"{AVERAGE.name} restart: {len(report.accepted)} jobs, {len(report.unavailable())} x 503, "
              f"worker retries={worker.retries}")

        assert report.statuses() <= {200, 201, 202, 503}
        assert report.outside_windows(report.unavailable()) == []
        _assert_users_recovered(report)
