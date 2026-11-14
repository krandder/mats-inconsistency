#!/usr/bin/env bash
# E37-E39: on-demand L4 launch across US and non-US L4 zones (the coordinator's fallback 1), retried every 20 min (up to 8
# attempts) until a VM is created. Usage: retry_jobs.sh <run-id-prefix> <jobfile> <note>; attempt i uses run id <prefix>i.
cd /home/agent/workspaces/mats/work/cm-interp-B
export GPU_ZONES="us-central1-a us-central1-b us-central1-c us-east4-a us-east4-c us-east1-b us-east1-c us-east1-d us-west1-a us-west1-b us-west1-c us-west4-a us-west4-c europe-west4-a europe-west4-b europe-west4-c europe-west1-b europe-west1-c europe-west3-a europe-west2-a asia-east1-a asia-east1-c asia-southeast1-a asia-southeast1-b asia-southeast1-c asia-northeast1-a asia-northeast1-c"
PFX=$1; JOBFILE=$2; NOTE=$3
for i in $(seq 1 8); do
  RID=$PFX$i
  ./gpu/launch_jobs.sh $RID $JOBFILE "$NOTE" > /tmp/cmb_launch_$RID.log 2>&1
  if grep -q "READY" /tmp/l4runs/$RID/gpu-run.out 2>/dev/null; then
    echo "$(date -u +%FT%TZ) attempt $i ($RID): VM was created and the job ran; stopping retries" >> GPU-LOG.md
    break
  fi
  echo "- $RID: no VM created (stockout in all listed zones); \$0." >> GPU-LOG.md
  sleep 1200
done
