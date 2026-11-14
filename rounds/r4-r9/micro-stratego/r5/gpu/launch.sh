#!/bin/bash
# launch.sh SEED [SPOT|STANDARD] [SUFFIX] : one named L4 rental for one seed through infra/gpu-run.sh (budget group r5,
# group cap $22, GPU_RUN_TIMEOUT=${TMO:-150m}, minimal startup) with gpu/drive.sh as the driver (MODE=full|smoke env).
# Runs detached; log /tmp/r5e2/launch-<run>.log; every launch is appended to gpu/launches.log.
SEED=$1
PROV=${2:-SPOT}
R5=/home/agent/workspaces/mats/work/micro-stratego/r5
INFRA=/home/agent/workspaces/eurisko-study/logical-soup/infra
SRC=/tmp/r5e2/src
[[ -d $SRC/data ]] || { echo "staging dir $SRC missing" >&2; exit 2; }
RUN=r5e2-s$SEED${3:-}
MODE=${MODE:-full}
TMO=${TMO:-150m}
echo "$(date -u +%FT%TZ) LAUNCH run=$RUN seed=$SEED mode=$MODE prov=$PROV timeout=$TMO" | tee -a $R5/gpu/launches.log
GPU_BUDGET_GROUP=r5 GPU_GROUP_BUDGET_USD=22 GPU_RUN_TIMEOUT=$TMO GPU_STARTUP=minimal GPU_PROVISIONING=$PROV \
GPU_SOURCE_DIR=$SRC setsid nohup $INFRA/gpu-run.sh --up --run-id $RUN --driver $R5/gpu/drive.sh -- $SEED $MODE \
  > /tmp/r5e2/launch-$RUN.log 2>&1 < /dev/null &
echo "pid $!"
