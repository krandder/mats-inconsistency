#!/bin/bash
# Local driver for logical-soup/infra/gpu-run.sh --driver. Args (after --): KIND SEED [RL] [SFT] [NST]
# Runs e3_remote.sh on the VM, then copies back only JSON/JSONL/CSV/PNG/TXT/LOG (no checkpoints)
# to /tmp/lir-agent/e3/runs (all) and the experiment dir (json/jsonl/csv/png/txt only).
set -u
source /home/agent/workspaces/eurisko-study/logical-soup/infra/lib.sh
KIND=$1; S=$2; shift 2
TMPR=/tmp/lir-agent/e3/runs; DST=/home/agent/workspaces/eurisko-study/lir-agent/experiments/E3/runs
mkdir -p $TMPR $DST
vm_ssh "$GPU_VM_IP" "sudo DEADLINE_MIN=${DEADLINE_MIN:-104} timeout --kill-after=30 108m bash /root/soup/E3/e3_remote.sh $KIND $S $*" > $TMPR/remote-$KIND-s$S.log 2>&1
rc=$?
rsync -az --rsync-path='sudo rsync' -e "ssh ${SSH_OPTS[*]}" --exclude='*.pt' "$SSH_USER@$GPU_VM_IP:/root/runs/E3/" "$TMPR/" || echo "sync failed" >&2
rsync -a --include='*/' --include='*.json' --include='*.jsonl' --include='*.csv' --include='*.png' --include='*.txt' --exclude='*' "$TMPR/" "$DST/"
echo "remote rc=$rc"
exit 0
