#!/bin/bash
# launch.sh [RUN-SUFFIX] : one on-demand (STANDARD) L4 for all three seeds through infra/gpu-run.sh (group r7, group cap
# $6.5, GPU_RUN_TIMEOUT=${TMO:-60m} so the worst-case reservation (1.6 h x $0.87 = $1.39) stays under the $1.50 ceiling,
# minimal startup), driver r7/gpu/drive.sh. Detached; log /tmp/r7e3/launch-<run>.log.
R7=/home/agent/workspaces/mats/work/micro-stratego/r7
INFRA=/home/agent/workspaces/eurisko-study/logical-soup/infra
[[ -d /tmp/r7e3/src/ckpt && -d /tmp/r7e3/src/data && -f /tmp/r7e3/src/r7/run_r7e3.py ]] || { echo "staging dir /tmp/r7e3/src incomplete" >&2; exit 2; }
RUN=r7e3-main${1:-}
TMO=${TMO:-60m}
echo "$(date -u +%FT%TZ) LAUNCH run=$RUN prov=STANDARD timeout=$TMO" | tee -a $R7/gpu/launches.log
GPU_BUDGET_GROUP=r7 GPU_GROUP_BUDGET_USD=6.5 GPU_RUN_TIMEOUT=$TMO GPU_STARTUP=minimal GPU_PROVISIONING=STANDARD \
GPU_SOURCE_DIR=/tmp/r7e3/src setsid nohup $INFRA/gpu-run.sh --up --run-id $RUN --driver $R7/gpu/drive.sh -- main \
  > /tmp/r7e3/launch-$RUN.log 2>&1 < /dev/null &
echo "pid $!"
