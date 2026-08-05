#!/usr/bin/env bash
# Compact view of what the Celery workers are doing right now.
#
#   ./scripts/queue-status.sh          one-shot
#   ./scripts/queue-status.sh -w       refresh every 3s
#
# Reads three sources, because none of them alone tells the whole story:
#   - Redis DB 2 (broker)   -> how many task messages are waiting per queue
#   - Redis DB 1 (metadata) -> the user-facing job records the UI polls
#   - Celery control        -> which task each worker process has in hand
#
# `celery inspect active` is not used for the running task: its output embeds
# the full task envelope (every audio asset), which is ~40 KB for a 144-file
# batch. The job records in DB 1 carry the same information in a readable form.

set -uo pipefail

REDIS=${REDIS_CONTAINER:-echo-redis}
BROKER_DB=2
JOB_DB=1
PYTHON_CMD=()

if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=(python3)
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD=(python)
elif command -v py >/dev/null 2>&1; then
  PYTHON_CMD=(py -3)
else
  echo "No Python interpreter found in PATH." >&2
  exit 1
fi

render() {
  echo "=== queue depth (messages waiting) ============================"
  for q in cpu gpu-fast gpu-large; do
    printf '  %-10s %s\n' "$q" "$(docker exec "$REDIS" redis-cli -n $BROKER_DB llen "$q" 2>/dev/null)"
  done

  local unacked
  unacked=$(docker exec "$REDIS" redis-cli -n $BROKER_DB hlen unacked 2>/dev/null)
  echo "  in-flight  ${unacked:-0}   (reserved by a worker, not yet acked)"

  echo
  echo "=== jobs ======================================================"
  printf '  %-16s %-14s %6s %-12s %s\n' OPERATION MODEL FILES STATUS PROGRESS

  # One exec that KEYS + MGETs inside the container: a docker exec per key costs
  # ~80 ms of round trip and there can be dozens. Plain `keys` (not --no-raw)
  # prints one bare key per line -- --no-raw pads indices with a leading space,
  # which silently drops entries 1-9 from any `sed 's/^[0-9]*) //'` cleanup.
  docker exec "$REDIS" sh -c "
    redis-cli -n $JOB_DB keys 'job:*' | grep -v completed-items > /tmp/jk
    [ -s /tmp/jk ] && xargs -a /tmp/jk redis-cli -n $JOB_DB mget
  " 2>/dev/null \
    | "${PYTHON_CMD[@]}" -c '
import sys, json

RANK = {"processing": 0, "started": 1, "queued": 2, "failure": 3, "cancelled": 4, "success": 5}
rows = []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        d = json.loads(line)
    except ValueError:
        continue
    p = d.get("progress") or {}
    progress = "%s/%s" % (p.get("current", 0), p.get("total", 0))
    rows.append((
        RANK.get(d.get("status"), 9),
        d.get("operation", "?"),
        str(d.get("model")),
        len(d.get("audio_ids") or []),
        d.get("status", "?"),
        progress,
        d.get("job_id", "")[:8],
    ))

for rank, op, model, n, status, prog, jid in sorted(rows):
    mark = "->" if rank == 0 else "  "
    print(f"  {mark} {op:<14} {model:<14} {n:>5} {status:<12} {prog:<9} {jid}")
if not rows:
    print("  (no job records — they expire after JOB_TTL_SECONDS)")
'

  echo
  echo "=== workers ==================================================="
  # NOTE: the heartbeat key should end in a worker hostname, but
  # publish_worker_heartbeat() in app/worker/tasks.py receives a Heart object
  # with no .hostname and falls back to str(sender) -- so the key carries an
  # object repr with a memory address. Count them rather than print them, and
  # read the real identities from Celery.
  local beats
  beats=$(docker exec "$REDIS" redis-cli -n $JOB_DB keys 'worker-heartbeat:*' 2>/dev/null | grep -c .)
  echo "  heartbeats: ${beats:-0}"
  docker exec "$REDIS" redis-cli -n $BROKER_DB client list 2>/dev/null >/dev/null
  printf '  consumers:  '
  docker exec echo-worker-model-local \
    celery -A app.core.celery_app:celery_app inspect ping --timeout 5 2>/dev/null \
    | grep -o 'celery@[a-z0-9]*' | paste -sd' ' - || echo '(unreachable)'
}

if [ "${1:-}" = "-w" ]; then
  while true; do clear; date; echo; render; sleep 3; done
else
  render
fi
