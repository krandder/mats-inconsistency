#!/bin/bash
# Local driver for logical-soup/infra/gpu-run.sh --driver (runs on General). Args: REMOTE_MINUTES JOBFILE
# Uploads checkpoints from /tmp/swarm-predictor/ckpt, starts remote.sh with the job file, and every 5 minutes syncs:
#   compact results (*.json, *.jsonl, *.txt, *.log)  -> li-net/swarm-predictor/runs/   (on /srv)
#   chain texts                                       -> /tmp/swarm-predictor/text/
#   LoRA checkpoints (*.pt)                           -> /tmp/swarm-predictor/ckpt/
# The VM is deleted by gpu-run.sh when this script exits.
set -u
MIN=$1; JOBS=$2
S=/home/agent/workspaces/eurisko-study/li-net/swarm-predictor
T=/tmp/swarm-predictor
source /home/agent/workspaces/eurisko-study/logical-soup/infra/lib.sh
RSH="ssh ${SSH_OPTS[*]}"
R="$SSH_USER@$GPU_VM_IP"
mkdir -p $S/runs $T/text $T/ckpt $T/logs
LOG=$T/logs/driver-$GPU_VM_NAME.log
vm_ssh "$GPU_VM_IP" "sudo mkdir -p /root/ckpt" >> $LOG 2>&1
rsync -az --rsync-path='sudo rsync' -e "$RSH" $T/ckpt/ "$R:/root/ckpt/" >> $LOG 2>&1 || echo "ckpt upload failed" >> $LOG
vm_ssh "$GPU_VM_IP" "sudo timeout --kill-after=60 ${MIN}m bash /root/soup/remote.sh $JOBS" >> $LOG 2>&1 &
rpid=$!
sync_all() {
  rsync -az --rsync-path='sudo rsync' -e "$RSH" --include='*/' --include='*.json' --include='*.jsonl' --include='*.txt' \
    --include='*.log' --exclude='*' "$R:/root/runs/swarm/" $S/runs/ >> $LOG 2>&1
  rsync -az --rsync-path='sudo rsync' -e "$RSH" "$R:/root/text/" $T/text/ >> $LOG 2>&1
  rsync -az --rsync-path='sudo rsync' -e "$RSH" "$R:/root/ckpt/" $T/ckpt/ >> $LOG 2>&1
}
while kill -0 $rpid 2>/dev/null; do
  for i in $(seq 1 30); do kill -0 $rpid 2>/dev/null || break; sleep 10; done
  sync_all
  echo "$(date -u +%FT%TZ) synced" >> $LOG
done
wait $rpid; rc=$?
sync_all; sync_all
echo "$(date -u +%FT%TZ) remote rc=$rc, final sync done" >> $LOG
exit 0
