#!/bin/bash
# mkvm.sh NAME : spot g2-standard-4 + 1x L4, termination DELETE, max-run 6 h, 100 GB boot disk (the image needs >= 100 GB) (auto-delete),
# apt timers masked by the startup script; tries the 13 L4 zones in order; prints "NAME ZONE" on success.
source /home/agent/workspaces/calibrated-ai/trader-economy/puzzles/P13/gpu/common.sh
N=$1
# spread slots over zones (22:40Z, after correlated spot preemptions in us-central1); from 23:00Z the 7 US zones outside
# us-central1 come first (rotated by slot number), then us-central1, then europe
ZA=($ZONES); k=$(echo $N | sed -E 's/^trader-b3-p13-([0-9]+).*/\1/'); [[ "$k" =~ ^[0-9]+$ ]] || k=0; k=$((k % 7))
ZO=(${ZA[@]:3:7})
ZR="${ZO[@]:$k:$((7-k))} ${ZO[@]:0:$k} ${ZA[@]:0:3} ${ZA[@]:10}"
for Z in ${ZONES_FIRST:-} $ZR; do
  if gcloud compute instances create $N $PROJ --zone $Z --machine-type g2-standard-4 --provisioning-model SPOT \
      --instance-termination-action DELETE --max-run-duration 21600s \
      --image pytorch-2-9-cu129-ubuntu-2204-nvidia-580-v20260909 --image-project deeplearning-platform-release \
      --boot-disk-size 100GB --boot-disk-type pd-balanced --boot-disk-auto-delete \
      --metadata-from-file startup-script=$G/startup.sh --metadata install-nvidia-driver=True \
      --labels exp=trader-p13,owner=trader-economy > $TMPD/mkvm_$N.log 2>&1; then
    echo "$N $Z"; exit 0; fi
  tail -1 $TMPD/mkvm_$N.log | cut -c1-140 >&2
done
exit 1
