#!/bin/bash
# mkvm.sh NAME : spot g2-standard-4 + 1x L4, termination DELETE, max-run 6 h, 100 GB boot disk (auto-delete); zones rotated by
# slot number (copied from P13/gpu/mkvm.sh); prints "NAME ZONE" on success.
source /home/agent/workspaces/calibrated-ai/trader-economy/puzzles/P18-detached-mazes/gpu/common.sh
N=$1
ZA=($ZONES); k=$(echo $N | sed -E 's/^trader-dds-([0-9]+).*/\1/'); [[ "$k" =~ ^[0-9]+$ ]] || k=0; k=$((k % ${#ZA[@]}))
ZR="${ZA[@]:$k} ${ZA[@]:0:$k}"
for Z in ${ZONES_FIRST:-} $ZR; do
  if gcloud compute instances create $N $PROJ --zone $Z --machine-type g2-standard-4 --provisioning-model SPOT \
      --instance-termination-action DELETE --max-run-duration 21600s \
      --image pytorch-2-9-cu129-ubuntu-2204-nvidia-580-v20260909 --image-project deeplearning-platform-release \
      --boot-disk-size 100GB --boot-disk-type pd-balanced --boot-disk-auto-delete \
      --metadata-from-file startup-script=$G/startup.sh --metadata install-nvidia-driver=True \
      --labels exp=trader-p18,owner=trader-economy > $TMPD/mkvm_$N.log 2>&1; then
    echo "$N $Z"; exit 0; fi
  tail -1 $TMPD/mkvm_$N.log | cut -c1-140 >&2
done
exit 1
