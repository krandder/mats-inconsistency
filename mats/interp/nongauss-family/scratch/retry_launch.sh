#!/bin/bash
# retry a GPU launch until a VM is created (stockouts); args: RID JOBSCRIPT MAXTRIES
cd /home/agent/workspaces/mats/work/nongauss-family
RID=$1; JOB=$2; MAX=${3:-12}
export GPU_ZONES="us-central1-c us-central1-a us-central1-b us-west1-a us-west1-b us-east4-a us-east4-c us-east1-c us-east1-d us-west4-a europe-west4-a europe-west4-b"
for i in $(seq 1 $MAX); do
  scratch/launch_gpu.sh ${RID}t$i "bash $JOB ${RID}t$i" > logs/launch_${RID}t$i.log 2>&1
  if grep -q "READY:" /tmp/l4runs/${RID}t$i/gpu-run.out 2>/dev/null; then echo "ran as ${RID}t$i"; exit 0; fi
  sleep 120
done
echo "gave up"
