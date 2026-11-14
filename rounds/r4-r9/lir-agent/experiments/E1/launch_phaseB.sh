#!/bin/bash
# Waits for the phase-A warm-start checkpoints, stages harness + e1 + fused warm ckpts, then launches V1..V6
# through gpu-run.sh as group admission allows (retry every 4 min; gpu-up refuses without creating anything).
E=/home/agent/workspaces/eurisko-study/lir-agent/experiments/E1
LS=/home/agent/workspaces/eurisko-study/logical-soup
until [ -f /tmp/lir-agent/E1/ckpt/warm-fused-s0/warm.pt ] && [ -f /tmp/lir-agent/E1/ckpt/warm-fused-s1/warm.pt ] && [ -f /tmp/lir-agent/E1/ckpt/warm-fused-s2/warm.pt ]; do sleep 60; done
S=/tmp/lir-agent/E1/src-B; mkdir -p $S/e1/ckpt
rsync -a /tmp/lir-agent/E1/src-A/ $S/; cp $E/e1run.py $E/e1_metrics.py $E/run_remote.sh $S/e1/; cp $E/jobs/* $S/e1/jobs/
for s in 0 1 2; do mkdir -p $S/e1/ckpt/warm-fused-s$s; cp /tmp/lir-agent/E1/ckpt/warm-fused-s$s/warm.pt $S/e1/ckpt/warm-fused-s$s/; done
cd $LS
for v in V1 V2 V3 V4 V5 V6; do
  id=lir-e1-$(echo $v | tr A-Z a-z)
  while true; do
    sp=$(bash -c 'source infra/lib.sh; spend_usd lir-agent')
    if awk -v s="$sp" 'BEGIN{exit !(s+5.22<=50)}'; then
      echo "$(date -u +%FT%TZ) launch $v spend=$sp"
      GPU_SOURCE_DIR=$S GPU_BUDGET_GROUP=lir-agent GPU_GROUP_BUDGET_USD=50 GPU_RUN_TIMEOUT=110m \
        setsid nohup infra/gpu-run.sh --up --run-id $id --driver $E/driver.sh -- $v > /tmp/lir-agent/E1/launch-$v.log 2>&1 < /dev/null &
      sleep 90
      grep -q REFUSE /tmp/lir-agent/E1/launch-$v.log && { echo "refused $v, retry"; sleep 20; continue; }
      break
    fi
    sleep 20
  done
done
echo "$(date -u +%FT%TZ) all launched"
