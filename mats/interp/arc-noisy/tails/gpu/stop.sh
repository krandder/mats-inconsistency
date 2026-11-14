#!/usr/bin/env bash
# stop.sh <run-id> : ask the job loop to exit (the VM is then deleted by gpu-run.sh cleanup)
RID=$1; HERE=/home/agent/workspaces/mats/work/arc-noisy
read IP KEY < <(bash $HERE/tails/gpu/vm.sh $RID) || exit 1
ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o LogLevel=ERROR soup@$IP "sudo touch /root/jobs/STOP" && echo "STOP sent"
echo "$(date -u +%FT%TZ) STOP rid=$RID" >> $HERE/GPU-LOG.md
