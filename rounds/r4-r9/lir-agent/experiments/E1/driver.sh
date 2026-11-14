#!/bin/bash
# Local driver for logical-soup/infra/gpu-run.sh --driver. Arg: job file name (e1/jobs/<name>).
# Syncs back: *.json/*.jsonl/*.csv/*.png -> experiments/E1/runs/ ; logs + eval pickles -> /tmp/lir-agent/E1/vm/<job>/ ;
# warm-start checkpoints -> /tmp/lir-agent/E1/ckpt/.
set -u
JOB=$1
E=/home/agent/workspaces/eurisko-study/lir-agent/experiments/E1
source /home/agent/workspaces/eurisko-study/logical-soup/infra/lib.sh
mkdir -p "$E/runs" /tmp/lir-agent/E1/vm/$JOB /tmp/lir-agent/E1/ckpt
vm_ssh "$GPU_VM_IP" "sudo timeout --kill-after=30 105m bash /root/soup/e1/run_remote.sh $JOB" > /tmp/lir-agent/E1/vm/$JOB/remote.log 2>&1
rc=$?
R="ssh ${SSH_OPTS[*]}"
rsync -az --rsync-path='sudo rsync' -e "$R" --exclude=ckpt --include='*/' --include='*.json' --include='*.jsonl' --include='*.csv' --include='*.png' --exclude='*' "$SSH_USER@$GPU_VM_IP:/root/runs/E1/" "$E/runs/" || echo "sync1 failed" >&2
rsync -az --rsync-path='sudo rsync' -e "$R" --exclude=ckpt "$SSH_USER@$GPU_VM_IP:/root/runs/E1/" "/tmp/lir-agent/E1/vm/$JOB/" || echo "sync2 failed" >&2
rsync -az --rsync-path='sudo rsync' -e "$R" --include='warm-*/' --include='warm.pt' --exclude='*' "$SSH_USER@$GPU_VM_IP:/root/runs/ckpt/" "/tmp/lir-agent/E1/ckpt/" || echo "sync3 failed" >&2
python3 -c "import json,sys; print(json.dumps(open(sys.argv[1]).read().splitlines(), indent=0))" /tmp/lir-agent/E1/vm/$JOB/jobs.log > "$E/runs/jobs-$JOB.json" 2>/dev/null
echo "remote rc=$rc"
exit 0
