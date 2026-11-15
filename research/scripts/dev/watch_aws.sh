#!/usr/bin/env bash
# Local watcher: exits (printing the line) when <logfile> on the research host contains <regex>,
# or after 3 consecutive SSH failures. Configure FALCON_REMOTE_HOST,
# FALCON_REMOTE_USER, FALCON_REMOTE_PATH, and optionally FALCON_SSH_KEY/FALCON_SSH_PORT.
set -euo pipefail
: "${FALCON_REMOTE_HOST:?Set FALCON_REMOTE_HOST to the research host name or address}"
: "${FALCON_REMOTE_USER:?Set FALCON_REMOTE_USER to the remote login name}"
: "${FALCON_REMOTE_PATH:?Set FALCON_REMOTE_PATH to the remote research directory}"
: "${1:?Usage: $0 <logfile> <regex>}"
: "${2:?Usage: $0 <logfile> <regex>}"
FALCON_SSH_PORT="${FALCON_SSH_PORT:-22}"
HOST="${FALCON_REMOTE_USER}@${FALCON_REMOTE_HOST}"
REMOTE_LOG="${FALCON_REMOTE_PATH%/}/${1#/}"
SSH_OPTS=(-p "$FALCON_SSH_PORT" -o BatchMode=yes -o ConnectTimeout=20)
if [ -n "${FALCON_SSH_KEY:-}" ]; then
  SSH_OPTS+=(-i "$FALCON_SSH_KEY")
fi
fails=0
while true; do
  if out=$(ssh "${SSH_OPTS[@]}" "$HOST" \
           "grep -E '$2' '$REMOTE_LOG' | tail -3" 2>&1); then
    fails=0
    [ -n "$out" ] && { echo "$1: $out"; exit 0; }
  else
    fails=$((fails+1)); [ $fails -ge 3 ] && { echo "SSH FAILING: $out"; exit 1; }
  fi
  sleep 300
done
