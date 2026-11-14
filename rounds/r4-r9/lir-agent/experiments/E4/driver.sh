#!/bin/bash
# Local driver for logical-soup/infra/gpu-run.sh --driver. Env: E4_KIND E4_PRESET E4_SEED.
# Syncs back only json/jsonl/log/txt (no pickles, no checkpoints).
set -u
source /home/agent/workspaces/eurisko-study/logical-soup/infra/lib.sh
R=/home/agent/workspaces/eurisko-study/lir-agent/experiments/E4/runs
mkdir -p $R/$E4_PRESET
vm_ssh "$GPU_VM_IP" "sudo timeout --kill-after=30 108m bash /root/soup/E4/remote.sh $E4_KIND $E4_PRESET $E4_SEED" > $R/$E4_PRESET/remote-$E4_KIND-s$E4_SEED.log 2>&1
rc=$?
rsync -az --rsync-path='sudo rsync' -e "ssh ${SSH_OPTS[*]}" --include='*/' --include='*.json' --include='*.jsonl' \
  --include='*.log' --include='*.txt' --exclude='*' "$SSH_USER@$GPU_VM_IP:/root/runs/E4/$E4_PRESET/" "$R/$E4_PRESET/" || echo "sync failed" >&2
echo "remote rc=$rc"
exit 0
