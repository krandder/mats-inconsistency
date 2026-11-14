#!/bin/bash
# Launch E2 VMs in priority order, one at a time, each only when the lir-agent group's GPU-LOG total
# (spent + 6-h reservations) + 5.22 <= 50 (gpu-up.sh enforces the same cap and refuses otherwise).
# Gives up on a VM after DEADLINE (UTC epoch) and records it as not launched.
set -u
INF=/home/agent/workspaces/eurisko-study/logical-soup/infra
E2=/home/agent/workspaces/eurisko-study/lir-agent/experiments/E2
LOG=/tmp/lir-agent/E2-launch
DEADLINE=${DEADLINE:-$(( $(date +%s) + 6*3600 ))}
mkdir -p $LOG
for i in "$@"; do
  while true; do
    s=$(bash -c "source $INF/lib.sh; spend_usd lir-agent" 2>/dev/null)
    if python3 -c "import sys; sys.exit(0 if float('$s')+5.22<=50 else 1)" 2>/dev/null; then
      JOBFILE=vm$i.txt GPU_SOURCE_DIR=/tmp/lir-agent/E2-src GPU_BUDGET_GROUP=lir-agent GPU_GROUP_BUDGET_USD=50 \
        GPU_RUN_TIMEOUT=115m setsid nohup $INF/gpu-run.sh --up --run-id lir-e2-vm$i --driver $E2/driver.sh -- vm$i \
        > $LOG/vm$i.log 2>&1 < /dev/null &
      sleep 75
      if grep -q REFUSE $LOG/vm$i.log; then echo "$(date -u +%FT%TZ) vm$i refused, retry"; sleep 30; continue; fi
      echo "$(date -u +%FT%TZ) vm$i launched (spend before $s)"; break
    fi
    [ $(date +%s) -gt $DEADLINE ] && { echo "$(date -u +%FT%TZ) vm$i NOT launched: deadline"; break; }
    sleep 10
  done
done
echo "$(date -u +%FT%TZ) launcher done"
