#!/bin/bash
# Install (or with --remove, uninstall) the gpu-watchdog user crontab entry, every 15 min.
# Idempotent: the entry is identified by the trailing marker comment.
set -eu
INFRA="$(cd "$(dirname "$0")" && pwd)"
MARK="# logical-soup-gpu-watchdog"
LINE="*/15 * * * * HOME=/home/agent PATH=/snap/bin:/usr/local/bin:/usr/bin:/bin $INFRA/gpu-watchdog.sh >>$INFRA/state/watchdog.log 2>&1 $MARK"
mkdir -p "$INFRA/state"
current=$(crontab -l 2>/dev/null || true)
filtered=$(grep -vF "$MARK" <<<"$current" || true)
if [[ ${1:-} == --remove ]]; then
  printf '%s\n' "$filtered" | crontab -
  echo "removed"; exit 0
fi
{ [[ -n $filtered ]] && printf '%s\n' "$filtered"; printf '%s\n' "$LINE"; } | crontab -
crontab -l | grep -F "$MARK"
