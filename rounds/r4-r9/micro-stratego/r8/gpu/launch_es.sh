#!/bin/bash
# launch.sh [RUN-SUFFIX] : one on-demand (STANDARD) L4 for all three seeds through infra/gpu-run.sh (group r8, group cap
# $2.2, GPU_RUN_TIMEOUT=${TMO:-30m} so the VM lives at most about 0.55 h (about $0.48, under the $0.50 ceiling),
# minimal startup), driver r8/gpu/drive.sh. Detached; log /tmp/r8e3/launch-<run>.log.
R8=/home/agent/workspaces/mats/work/micro-stratego/r8
INFRA=/home/agent/workspaces/eurisko-study/logical-soup/infra
[[ -d /tmp/r8e3/src/ckpt && -d /tmp/r8e3/src/data && -f /tmp/r8e3/src/r8/expl_d1_es.py ]] || { echo "staging dir /tmp/r8e3/src incomplete" >&2; exit 2; }
RUN=r8e3-es${1:-}
TMO=${TMO:-25m}
echo "$(date -u +%FT%TZ) LAUNCH run=$RUN prov=STANDARD timeout=$TMO" | tee -a $R8/gpu/launches.log
GPU_BUDGET_GROUP=r8 GPU_GROUP_BUDGET_USD=2.2 GPU_RUN_TIMEOUT=$TMO GPU_STARTUP=minimal GPU_PROVISIONING=STANDARD \
GPU_SOURCE_DIR=/tmp/r8e3/src setsid nohup $INFRA/gpu-run.sh --up --run-id $RUN --driver $R8/gpu/drive_es.sh -- main \
  > /tmp/r8e3/launch-$RUN.log 2>&1 < /dev/null &
echo "pid $!"
