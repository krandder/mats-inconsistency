#!/bin/bash
# Install (or with --remove, uninstall) the two traders-net cron entries:
#   every 2 min  queue-runner.sh --once   (dispatch)
#   every 15 min gpu-watchdog.sh          (stop guarantee, cost audit, COST.md)
# Idempotent: entries are identified by their trailing marker comments.
set -eu
INFRA="$(cd "$(dirname "$0")" && pwd)"
ENV="HOME=/home/agent PATH=/snap/bin:/usr/local/bin:/usr/bin:/bin:/home/agent/.local/bin"
L1="*/2 * * * * $ENV $INFRA/queue-runner.sh --once >>$INFRA/state/runner.log 2>&1 # traders-net-gpu-queue"
L2="*/15 * * * * $ENV $INFRA/gpu-watchdog.sh >>$INFRA/state/watchdog.log 2>&1 # traders-net-gpu-watchdog"
mkdir -p "$INFRA/state"
cur=$(crontab -l 2>/dev/null || true)
keep=$(grep -vE '# traders-net-gpu-(queue|watchdog)$' <<<"$cur" || true)
if [[ ${1:-} == --remove ]]; then printf '%s\n' "$keep" | crontab -; echo removed; exit 0; fi
{ [[ -n $keep ]] && printf '%s\n' "$keep"; printf '%s\n%s\n' "$L1" "$L2"; } | crontab -
crontab -l | grep -E '# traders-net-gpu-(queue|watchdog)$'
