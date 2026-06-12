#!/usr/bin/env bash
set -euo pipefail

PORT=8080
INPUT_DIR="experiments/placeholders"
OUTPUT_JSON="leaderboard_site/data/leaderboard.json"

usage() {
  cat <<'EOF'
Usage:
  ./leaderboard_site/run_local_leaderboard.sh [--port 8080] [--input-dir experiments/placeholders]

Behavior:
  1) Build leaderboard data JSON
  2) Start a local static server from repo root

Open in browser:
  http://localhost:<port>/leaderboard_site/
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    --input-dir)
      INPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "[1/2] Building leaderboard JSON from ${INPUT_DIR} ..."
BUILD_CMD=(python leaderboard_site/scripts/build_leaderboard_data.py
  --input-dir "${INPUT_DIR}"
  --output "${OUTPUT_JSON}")

"${BUILD_CMD[@]}"

echo "[2/2] Serving at http://localhost:${PORT}/leaderboard_site/"
python -m http.server "${PORT}"
