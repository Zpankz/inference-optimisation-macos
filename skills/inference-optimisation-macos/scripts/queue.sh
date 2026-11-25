#!/usr/bin/env bash
# macOS-safe single-GPU benchmark serializer using POSIX atomic mkdir locking.
# Replaces Linux-only flock so overlapping agent commands never share unified GPU bandwidth.
# Uses atomic rename before stale-lock cleanup to prevent race conditions between waiters.
#
# Usage:
#   bash scripts/queue.sh <job-label> -- <command...>
#   QUEUE_LOCK_DIR=/tmp/my_gpu.lock QUEUE_TIMEOUT=600 bash scripts/queue.sh ab-1 -- python3 ...

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <job-label> -- <command...>" >&2
  exit 2
fi

LABEL="$1"
shift
if [[ "$1" != "--" ]]; then
  echo "Error: expected '--' after job label '$LABEL'" >&2
  exit 2
fi
shift

LOCK_DIR="${QUEUE_LOCK_DIR:-/tmp/apple_silicon_inference_gpu.lock}"
TIMEOUT="${QUEUE_TIMEOUT:-1800}"
POLL_INTERVAL="${QUEUE_POLL_INTERVAL:-1}"
PID_FILE="$LOCK_DIR/pid"
LABEL_FILE="$LOCK_DIR/label"

acquire_lock() {
  local start_ts now_ts elapsed holder_pid holder_label stale_dir
  start_ts="$(date +%s)"

  while true; do
    if mkdir "$LOCK_DIR" 2>/dev/null; then
      echo "$$" > "$PID_FILE"
      echo "$LABEL" > "$LABEL_FILE"
      return 0
    fi

    # Recover stale lock atomically via rename so concurrent waiters cannot delete a fresh lock
    if [[ -f "$PID_FILE" ]]; then
      holder_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
      if [[ -n "$holder_pid" ]] && ! kill -0 "$holder_pid" 2>/dev/null; then
        stale_dir="${LOCK_DIR}.stale.$$"
        if mv "$LOCK_DIR" "$stale_dir" 2>/dev/null; then
          echo "[queue] Reaped stale lock from dead PID $holder_pid" >&2
          rm -rf "$stale_dir" 2>/dev/null || true
          continue
        fi
      fi
    fi

    now_ts="$(date +%s)"
    elapsed=$((now_ts - start_ts))
    if [[ "$elapsed" -ge "$TIMEOUT" ]]; then
      holder_label="$(cat "$LABEL_FILE" 2>/dev/null || echo unknown)"
      echo "[queue] Timed out after ${TIMEOUT}s waiting for GPU lock (held by '$holder_label')" >&2
      return 1
    fi

    sleep "$POLL_INTERVAL"
  done
}

release_lock() {
  if [[ -f "$PID_FILE" ]] && [[ "$(cat "$PID_FILE" 2>/dev/null || true)" == "$$" ]]; then
    rm -rf "$LOCK_DIR" 2>/dev/null || true
  fi
}

trap release_lock EXIT INT TERM

acquire_lock
echo "[queue] Acquired GPU lock for '$LABEL' (pid=$$)" >&2
"$@"
