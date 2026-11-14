#!/bin/bash
# launch.sh [RUN-SUFFIX] : one on-demand (STANDARD) L4 for all three seeds through infra/gpu-run.sh (group r6, group cap
# $12, GPU_RUN_TIMEOUT=${TMO:-75m}, minimal startup), driver r6/gpu/drive.sh. Detached; log /tmp/r6e3/launch-<run>.log.
R6=/home/agent/workspaces/mats/work/micro-stratego/r6
INFRA=/home/agent/workspaces/eurisko-study/logical-soup/infra
[[ -d /tmp/r6e3/src/ckpt && -d /tmp/r6e3/src/data ]] || { echo "staging dir /tmp/r6e3/src incomplete" >&2; exit 2; }
RUN=r6e3-main${1:-}
TMO=${TMO:-75m}
echo "$(date -u +%FT%TZ) LAUNCH run=$RUN prov=STANDARD timeout=$TMO" | tee -a $R6/gpu/launches.log
GPU_BUDGET_GROUP=r6 GPU_GROUP_BUDGET_USD=12 GPU_RUN_TIMEOUT=$TMO GPU_STARTUP=minimal GPU_PROVISIONING=STANDARD \
GPU_SOURCE_DIR=/tmp/r6e3/src setsid nohup $INFRA/gpu-run.sh --up --run-id $RUN --driver $R6/gpu/drive.sh -- main \
  > /tmp/r6e3/launch-$RUN.log 2>&1 < /dev/null &
echo "pid $!"
