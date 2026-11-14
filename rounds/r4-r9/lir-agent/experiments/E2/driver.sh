#!/bin/bash
# Local driver for gpu-run.sh --driver. Env JOBFILE (e.g. vm1.txt). Runs remote.sh under a 105-min timeout,
# rsyncs only json/jsonl/csv/png from /root/runs/E2 every 8 min and at the end (checkpoints die with the VM).
set -u
E2=/home/agent/workspaces/eurisko-study/lir-agent/experiments/E2
source /home/agent/workspaces/eurisko-study/logical-soup/infra/lib.sh
DEST=$E2/runs/${JOBFILE%.txt}; mkdir -p $DEST
sync() { rsync -az --rsync-path='sudo rsync' -e "ssh ${SSH_OPTS[*]}" --include='*/' --include='*.json' --include='*.jsonl' \
  --include='*.csv' --include='*.png' --exclude='*' "$SSH_USER@$GPU_VM_IP:/root/runs/E2/" "$DEST/" 2>>$DEST/sync.err.txt; }
( while true; do sleep 480; sync; done ) & SP=$!
vm_ssh "$GPU_VM_IP" "sudo timeout --kill-after=30 105m bash /root/soup/e2/remote.sh $JOBFILE" > /tmp/lir-agent/E2-remote-$JOBFILE.log 2>&1
rc=$?
kill $SP 2>/dev/null
sync; sync
echo "remote rc=$rc"
exit 0
