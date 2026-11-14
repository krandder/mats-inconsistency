#!/bin/bash
# launch2.sh MODE: launch when the programme's actual-cost estimate (lir-agent/GUARD.log total_est =
# audited DELETE costs + running VMs at the rate ceiling, eurisko's guard) plus this VM's 2.2 $ worst
# case stays <= $50; group reservation cap passed as the $60 programme cap (as eurisko does).
set -u
MODE=$1
INF=/home/agent/workspaces/eurisko-study/logical-soup/infra
E=/home/agent/workspaces/eurisko-study/lir-agent/experiments/E7
G=/home/agent/workspaces/eurisko-study/lir-agent/GUARD.log
L=/tmp/lir-agent/e7/launch-$MODE.log
est=$(grep STATUS $G | tail -1 | grep -oE "total_est=[0-9.]+" | cut -d= -f2)
if awk -v s="$est" 'BEGIN{exit !(s+2.2<=50)}'; then
  echo "$(date -u +%FT%TZ) $MODE launching, GUARD total_est=$est (+2.2 worst case <= 50)" >> $L
  GPU_SOURCE_DIR=/tmp/lir-agent/e7/src GPU_BUDGET_GROUP=lir-agent GPU_GROUP_BUDGET_USD=60 GPU_RUN_TIMEOUT=115m \
    $INF/gpu-run.sh --up --run-id lir-e7-$MODE --driver $E/driver.sh -- $MODE >> $L 2>&1
  echo "$(date -u +%FT%TZ) $MODE finished rc=$?" >> $L
else
  echo "$(date -u +%FT%TZ) $MODE NOT RUN: GUARD total_est=$est + 2.2 > 50" >> $L
fi
