#!/usr/bin/env bash
# E21: try the on-demand L4 launch now and then every 20 min (up to 12 attempts) until a VM is created; US and non-US L4 zones
# (coordinator's fallback 1). The job packs 6 trainings per VM. Each attempt is logged in GPU-LOG.md.
cd /home/agent/workspaces/mats/work/cm-interp-B
export GPU_ZONES="us-central1-a us-central1-b us-central1-c us-east4-a us-east4-c us-east1-b us-east1-c us-east1-d us-west1-a us-west1-b us-west1-c us-west4-a us-west4-c europe-west4-a europe-west4-b europe-west4-c europe-west1-b europe-west1-c europe-west3-a europe-west2-a asia-east1-a asia-east1-c asia-southeast1-a asia-southeast1-b asia-southeast1-c asia-northeast1-a asia-northeast1-c"
for i in $(seq 3 14); do
  RID=cmb-e21-w$i
  ./gpu/launch_e21.sh $RID > /tmp/cmb_e21_launch$i.log 2>&1
  if grep -q "READY" /tmp/l4runs/$RID/gpu-run.out 2>/dev/null; then
    echo "$(date -u +%FT%TZ) attempt $i ($RID): VM was created and the job ran; stopping retries" >> GPU-LOG.md
    break
  fi
  echo "- $RID: no VM created (stockout in all listed zones); \$0." >> GPU-LOG.md
  sleep 1200
done
