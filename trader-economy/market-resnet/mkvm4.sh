#!/bin/bash
# spot g2-standard-4 + L4, termination DELETE (VM and disk go on preemption or max-run), max-run 4 h; tries zones; prints "name zone"
N=$1
for Z in ${ZONES:-us-west1-b us-east1-c us-west4-a us-east4-a us-west1-a us-east1-d us-east4-c us-central1-a us-central1-b us-central1-c europe-west4-a europe-west4-b europe-west1-b}; do
  if gcloud compute instances create $N --project GCP_PROJECT --zone $Z --machine-type g2-standard-4 --provisioning-model SPOT --instance-termination-action DELETE --max-run-duration 14400s --image pytorch-2-9-cu129-ubuntu-2204-nvidia-580-v20260909 --image-project deeplearning-platform-release --boot-disk-size 100GB --boot-disk-type pd-balanced --boot-disk-auto-delete --metadata install-nvidia-driver=True --labels exp=trader-mresnet,owner=trader-economy >/tmp/mrtest/mkvm_$N.log 2>&1; then
    echo "$N $Z $(date -u +%H:%M)"; exit 0; fi
  tail -1 /tmp/mrtest/mkvm_$N.log | cut -c1-120
done
exit 1
