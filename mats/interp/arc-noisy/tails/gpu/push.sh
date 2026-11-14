#!/usr/bin/env bash
# push.sh <run-id> : sync the current tails/*.py (+ anet.py, rtable.json) to /root/soup on the live VM
set -e
RID=$1; HERE=/home/agent/workspaces/mats/work/arc-noisy
read IP KEY < <(bash $HERE/tails/gpu/vm.sh $RID)
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o LogLevel=ERROR"
T=/tmp/arcnoisy_push/$RID; mkdir -p $T/tails
cp $HERE/anet.py $HERE/samplers.py $HERE/kerdock129.npy $HERE/rtable.json $T/; cp $HERE/tails/*.py $T/tails/; cp $HERE/tails/gpu/*.py $T/tails/ 2>/dev/null || true
mkdir -p $T/thr_eval; cp $HERE/tails/runs/thr_eval/*.json $T/thr_eval/ 2>/dev/null || true
rsync -az --rsync-path="sudo rsync" -e "$SSH" $T/ soup@$IP:/root/soup/
echo pushed
