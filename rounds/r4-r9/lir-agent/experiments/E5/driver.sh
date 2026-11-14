#!/bin/bash
# Local driver for gpu-run.sh --driver. Args: JOBFILE (path relative to the staged dir). Syncs
# JSON/JSONL/CSV/PNG/log from /root/runs/E5 every 5 min and at the end into experiments/E5/runs/.
set -u
X=/home/agent/workspaces/eurisko-study/lir-agent/experiments/E5
source /home/agent/workspaces/eurisko-study/logical-soup/infra/lib.sh
mkdir -p $X/runs
sync_back() { rsync -az --rsync-path='sudo rsync' -e "ssh ${SSH_OPTS[*]}" --include='*/' --include='*.json' --include='*.jsonl' \
  --include='*.csv' --include='*.png' --include='*.log' --exclude='*' "$SSH_USER@$GPU_VM_IP:/root/runs/E5/" "$X/runs/" 2>/dev/null; }
( while sleep 300; do sync_back; done ) & SP=$!
vm_ssh "$GPU_VM_IP" "sudo timeout --kill-after=30 ${E5_REMOTE_TIMEOUT:-105m} bash /root/soup/e5/remote.sh /root/soup/e5/$1" > "$X/runs/remote-$1.log" 2>&1
rc=$?
kill $SP 2>/dev/null
sync_back; sync_back
echo "remote rc=$rc"
exit 0
