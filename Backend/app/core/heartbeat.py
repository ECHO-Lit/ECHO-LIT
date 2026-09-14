"""Worker liveness, readable in O(log n) instead of by scanning the keyspace.

Workers used to write one `worker-heartbeat:{hostname}` key each, and /health
counted them with `KEYS worker-heartbeat:*`. KEYS walks every key in the job
database -- every job, audio and model record -- and blocks the Redis server
while it does, so the probe a monitor calls every few seconds grew with total
usage. Heartbeats now live in one sorted set scored by time.

Kept in app.core, not app.worker: /health must not import worker code, and the
worker writes through a synchronous client while the API reads through an async
one, so both helpers take the client as an argument.
"""

from __future__ import annotations

HEARTBEAT_KEY = "worker-heartbeats"
HEARTBEAT_FRESH_SECONDS = 90


def record_worker_heartbeat(client, hostname: str, now: float):
    """Record `hostname` as alive at `now` and drop long-dead entries.

    Returns the pipeline's `execute()` result -- a list for a sync client, an
    awaitable for an async one.
    """
    pipe = client.pipeline()
    pipe.zadd(HEARTBEAT_KEY, {hostname: now})
    # Bounds the set: a host that has not beaten for ten windows is gone.
    pipe.zremrangebyscore(HEARTBEAT_KEY, "-inf", now - 10 * HEARTBEAT_FRESH_SECONDS)
    pipe.expire(HEARTBEAT_KEY, 10 * HEARTBEAT_FRESH_SECONDS)
    return pipe.execute()


async def count_live_workers(client, now: float) -> int:
    return int(await client.zcount(HEARTBEAT_KEY, now - HEARTBEAT_FRESH_SECONDS, "+inf"))
