#!/usr/bin/env bash
# Stop the ECHO development stack without deleting Docker volumes or local data.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

docker compose down
echo "ECHO services stopped. Docker volumes were preserved."
