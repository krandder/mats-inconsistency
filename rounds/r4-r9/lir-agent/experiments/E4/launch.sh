#!/bin/bash
# launch.sh KIND PRESET SEED : one named L4 rental through gpu-run.sh (run-id lir-E4-...), budget group lir-agent.
set -u
KIND=$1; PRESET=$2; SEED=$3
E=/home/agent/workspaces/eurisko-study/lir-agent/experiments/E4
mkdir -p $E/runs/$PRESET
RID=lir-e4-$(echo $PRESET | tr -d '_')-$(echo $KIND | tr A-Z a-z)-s$SEED
cd /home/agent/workspaces/eurisko-study/logical-soup
E4_KIND=$KIND E4_PRESET=$PRESET E4_SEED=$SEED GPU_SOURCE_DIR=/tmp/lir-agent/E4/src GPU_BUDGET_GROUP=lir-agent \
GPU_GROUP_BUDGET_USD=50 GPU_RUN_TIMEOUT=115m \
  infra/gpu-run.sh --up --run-id "$RID" --driver $E/driver.sh -- x \
  > $E/runs/$PRESET/launch-$KIND-s$SEED.log 2>&1
echo "launch $KIND $PRESET $SEED rc=$?" >> $E/runs/launch.log
