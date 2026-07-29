#!/usr/bin/env bash
# Start the complete ECHO development stack and wait for the UI and API.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if ! docker info >/dev/null 2>&1; then
  echo "Docker Desktop is not running. Start it, then rerun this script." >&2
  exit 1
fi

docker compose up -d --build

wait_for_http() {
  local url="$1"
  local name="$2"

  for _ in $(seq 1 60); do
    if curl --fail --silent --show-error --max-time 2 "$url" >/dev/null; then
      echo "$name is ready: $url"
      return 0
    fi
    sleep 2
  done

  echo "$name did not become ready. Inspect logs with: docker compose logs --tail=100" >&2
  return 1
}

wait_for_http "http://localhost:8000/health" "API"
wait_for_http "http://localhost:8080" "Frontend"

echo
echo "Open ECHO at: http://localhost:8080"
docker compose ps
