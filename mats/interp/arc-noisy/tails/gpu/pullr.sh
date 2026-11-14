#!/usr/bin/env bash
# pullr.sh <run-id> : rsync /root/runs/<rid>/ from the live VM into tails/runs/<rid>/
RID=$1; HERE=/home/agent/workspaces/mats/work/arc-noisy
read IP KEY < <(bash $HERE/tails/gpu/vm.sh $RID) || exit 1
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o LogLevel=ERROR"
mkdir -p $HERE/tails/runs/$RID
timeout 600 rsync -az --rsync-path="sudo rsync" -e "$SSH" soup@$IP:/root/runs/$RID/ $HERE/tails/runs/$RID/
