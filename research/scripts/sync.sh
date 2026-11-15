#!/usr/bin/env bash
# Push code to / pull results from a research host.
# Configure FALCON_REMOTE_HOST, FALCON_REMOTE_USER, FALCON_REMOTE_PATH, and
# optionally FALCON_SSH_KEY and FALCON_SSH_PORT before running.
set -euo pipefail
: "${FALCON_REMOTE_HOST:?Set FALCON_REMOTE_HOST to the research host name or address}"
: "${FALCON_REMOTE_USER:?Set FALCON_REMOTE_USER to the remote login name}"
: "${FALCON_REMOTE_PATH:?Set FALCON_REMOTE_PATH to the remote research directory}"
FALCON_SSH_PORT="${FALCON_SSH_PORT:-22}"
HOST="${FALCON_REMOTE_USER}@${FALCON_REMOTE_HOST}"
REMOTE="${FALCON_REMOTE_PATH%/}/"
SSH_OPTS=(-p "$FALCON_SSH_PORT" -o BatchMode=yes)
if [ -n "${FALCON_SSH_KEY:-}" ]; then
  SSH_OPTS+=(-i "$FALCON_SSH_KEY")
fi
SSH="ssh ${SSH_OPTS[*]}"
LOCAL="$(cd "$(dirname "$0")/.." && pwd)/"
case "$1" in
  push) rsync -az -e "$SSH" --exclude experiments/ --exclude checkpoints/ --exclude datasets/hf_cache \
          --exclude external_repos/ --exclude .venv/ --exclude '__pycache__' "$LOCAL" "$HOST:$REMOTE" ;;
  pull) rsync -az -e "$SSH" --include '*/' --include '*.json' --include '*.jsonl' --include '*.csv' \
          --include '*.yaml' --include '*.log' --include '*.png' --include '*.txt' --exclude '*' \
          "$HOST:${REMOTE}experiments/" "${LOCAL}experiments/"
        rsync -az -e "$SSH" "$HOST:${REMOTE}external_repos/VERSIONS.txt" "${LOCAL}external_repos/" ;;
  *) echo "Usage: FALCON_REMOTE_HOST=... FALCON_REMOTE_USER=... FALCON_REMOTE_PATH=... $0 push|pull" >&2; exit 2 ;;
esac
