#!/bin/bash
# Local driver for logical-soup/infra/gpu-run.sh --driver (runs on General). Args: REMOTE_MINUTES JOBFILE PAR
# Starts remote.sh on the VM and every 2 minutes syncs result files (*.json, *.csv, *.png only)
# from /root/runs/eca/ to /tmp/eca/runs/. The VM is deleted by gpu-run.sh when this script exits.
set -u
MIN=$1; JOBS=$2; PAR=${3:-2}
T=/tmp/eca
source /home/agent/workspaces/eurisko-study/logical-soup/infra/lib.sh
RSH="ssh ${SSH_OPTS[*]}"
R="$SSH_USER@$GPU_VM_IP"
mkdir -p $T/runs $T/logs
LOG=$T/logs/driver-$GPU_VM_NAME.log
vm_ssh "$GPU_VM_IP" "sudo timeout --kill-after=60 ${MIN}m bash /root/soup/remote.sh $JOBS $PAR $MIN" >> $LOG 2>&1 &
rpid=$!
sync_all() {
  rsync -az --rsync-path='sudo rsync' -e "$RSH" --include='*/' --include='*.json' --include='*.jsonl' --include='*.csv' \
    --include='*.png' --include='*.pt' --exclude='*' "$R:/root/runs/eca/" $T/runs/ >> $LOG 2>&1
}
while kill -0 $rpid 2>/dev/null; do
  for i in $(seq 1 12); do kill -0 $rpid 2>/dev/null || break; sleep 10; done
  sync_all
  echo "$(date -u +%FT%TZ) synced $(find $T/runs -name 'result.json' | wc -l) result.json" >> $LOG
done
wait $rpid; rc=$?
sync_all; sync_all
echo "$(date -u +%FT%TZ) remote rc=$rc, final sync done" >> $LOG
exit 0
