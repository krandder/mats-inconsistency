#!/usr/bin/env bash
# job.sh <run-id> <name> '<command run in /root/soup>' : queue a job on the live VM (after pushing code)
set -e
RID=$1; NAME=$2; CMD=$3; HERE=/home/agent/workspaces/mats/work/arc-noisy
bash $HERE/tails/gpu/push.sh $RID >/dev/null
read IP KEY < <(bash $HERE/tails/gpu/vm.sh $RID)
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o LogLevel=ERROR"
F=/tmp/arcnoisy_push/$RID/$NAME.sh
printf 'export OUT=/root/runs/%s\nnvidia-smi -L\n%s\n' "$RID" "$CMD" > $F
rsync -az --rsync-path="sudo rsync" -e "$SSH" $F soup@$IP:/root/jobs/
echo "$(date -u +%FT%TZ) JOB rid=$RID name=$NAME cmd=\"${CMD:0:160}\"" >> $HERE/GPU-LOG.md
echo queued $NAME
