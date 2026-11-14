#!/bin/bash
# Local driver for gpu-run --driver (runs on General). Args: REMOTE_MINUTES JOBFILE PAR
# Starts remote.sh on the VM; every 60 s syncs /root/runs/la/ (json, jsonl, csv, pt, npz, log) to linear-agg/runs/.
set -u
MIN=$1; JOBS=$2; PAR=${3:-2}
LA=/home/agent/workspaces/mats/work/linear-agg
source /home/agent/workspaces/eurisko-study/logical-soup/infra/lib.sh
RSH="ssh ${SSH_OPTS[*]}"
R="$SSH_USER@$GPU_VM_IP"
mkdir -p $LA/runs $LA/gpu/logs
LOG=$LA/gpu/logs/driver-$GPU_VM_NAME.log
vm_ssh "$GPU_VM_IP" "sudo timeout --kill-after=60 ${MIN}m bash /root/soup/gpu/remote.sh $JOBS $PAR" >> $LOG 2>&1 &
rpid=$!
sync_all() {
  rsync -az --rsync-path='sudo rsync' -e "$RSH" --include='*/' --include='*.json' --include='*.jsonl' --include='*.csv' \
    --include='*.pt' --include='*.npz' --include='*.log' --exclude='*' "$R:/root/runs/la/" $LA/runs/ >> $LOG 2>&1
  rsync -az --rsync-path='sudo rsync' -e "$RSH" "$R:/root/logs/" $LA/runs/vmlogs-$GPU_VM_NAME/ >> $LOG 2>&1
}
while kill -0 $rpid 2>/dev/null; do
  for i in $(seq 1 6); do kill -0 $rpid 2>/dev/null || break; sleep 10; done
  sync_all
  echo "$(date -u +%FT%TZ) synced" >> $LOG
done
wait $rpid; rc=$?
sync_all; sync_all
echo "$(date -u +%FT%TZ) remote rc=$rc, final sync done" >> $LOG
exit 0
