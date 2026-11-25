#!/usr/bin/env bash
# Portable single-GPU benchmark serializer for macOS and Linux.
# Uses Python's fcntl.flock inside a private per-user 0700 directory with O_NOFOLLOW
# and ownership checks, so locks release automatically on crash/SIGKILL without stale-lock races.
#
# Usage:
#   bash scripts/queue.sh <job-label> -- <command...>
#   QUEUE_LOCK=/path/to/custom.lock bash scripts/queue.sh ab-1 -- python3 ...

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

exec python3 - "$LABEL" "$@" <<'PY'
import fcntl
import os
import stat
import subprocess
import sys
import tempfile

label = sys.argv[1]
cmd = sys.argv[2:]
uid = os.getuid()
custom = os.environ.get("QUEUE_LOCK")

if custom:
    lock_path = os.path.abspath(custom)
    parent = os.path.dirname(lock_path)
    os.makedirs(parent, mode=0o700, exist_ok=True)
else:
    base = os.environ.get("XDG_RUNTIME_DIR") or os.path.join(
        tempfile.gettempdir(), f"inference-optimisation-macos-{uid}"
    )
    try:
        os.makedirs(base, mode=0o700, exist_ok=True)
        st = os.stat(base, follow_symlinks=False)
    except OSError as exc:
        sys.exit(f"[queue] cannot create private lock directory {base}: {exc}")
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != uid or (stat.S_IMODE(st.st_mode) & 0o077):
        sys.exit(f"[queue] lock directory must be private (0700) and owned by uid {uid}: {base}")
    lock_path = os.path.join(base, "gpu.lock")

flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
try:
    fd = os.open(lock_path, flags, 0o600)
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode) or st.st_uid != uid:
        os.close(fd)
        sys.exit(f"[queue] refusing non-regular or foreign-owned lock file: {lock_path}")
except OSError as exc:
    sys.exit(f"[queue] failed to open lock file {lock_path}: {exc}")

fcntl.flock(fd, fcntl.LOCK_EX)
os.ftruncate(fd, 0)
os.write(fd, f"{os.getpid()} {label}\n".encode("utf-8"))
sys.stderr.write(f"[queue] Acquired GPU lock for '{label}' (pid={os.getpid()})\n")
sys.stderr.flush()

try:
    rc = subprocess.call(cmd)
finally:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)

sys.exit(rc)
PY
