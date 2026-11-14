#!/bin/bash
# launch.sh MODE: wait (up to 5 h) until the lir-agent group has room for one more L4 reservation
# under $50 (GPU-LOG spend_usd lir-agent, reservations included), then gpu-run.sh with driver.sh MODE.
set -u
MODE=$1
INF=/home/agent/workspaces/eurisko-study/logical-soup/infra
E=/home/agent/workspaces/eurisko-study/lir-agent/experiments/E7
L=/tmp/lir-agent/e7/launch-$MODE.log
end=$(( $(date +%s) + 5*3600 ))
while [ $(date +%s) -lt $end ]; do
  sp=$(bash -c "source $INF/lib.sh; spend_usd lir-agent")
  if awk -v s="$sp" 'BEGIN{exit !(s+2.20<=50)}'; then
    echo "$(date -u +%FT%TZ) $MODE launching, lir-agent spend+reserve=$sp" >> $L
    GPU_SOURCE_DIR=/tmp/lir-agent/e7/src GPU_BUDGET_GROUP=lir-agent GPU_GROUP_BUDGET_USD=50 GPU_RUN_TIMEOUT=115m \
      $INF/gpu-run.sh --up --run-id lir-e7-$MODE --driver $E/driver.sh -- $MODE >> $L 2>&1
    rc=$?
    if grep -q "REFUSE: group" <(tail -5 $L) && ! grep -q "RUN-END\|remote rc" <(tail -20 $L); then
      echo "$(date -u +%FT%TZ) refused, retry" >> $L; sleep $((5 + RANDOM % 5)); continue
    fi
    echo "$(date -u +%FT%TZ) $MODE finished rc=$rc" >> $L; exit $rc
  fi
  sleep $((5 + RANDOM % 5))
done
echo "$(date -u +%FT%TZ) $MODE NOT RUN: budget room never appeared within 5 h" >> $L; exit 9
