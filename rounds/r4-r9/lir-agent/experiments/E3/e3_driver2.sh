#!/bin/bash
# Local driver for logical-soup/infra/gpu-run.sh --driver. Args (after --): KIND SEED [more]
#   KIND roles|fused|rolestrain: e3_remote.sh KIND SEED [RL] [SFT] [NST]; rolestrain also copies checkpoints to /tmp.
#   KIND eval: e3_remote_eval.sh SEED [SEED ...] (checkpoints staged in the source dir under ck/).
# Copies back only JSON/JSONL/CSV/PNG/TXT/LOG (no checkpoints) to /tmp/lir-agent/e3/runs, and json/jsonl/csv/png/txt
# to the experiment dir.
set -u
source /home/agent/workspaces/eurisko-study/logical-soup/infra/lib.sh
KIND=$1; S=$2; shift 2
TMPR=/tmp/lir-agent/e3/runs; DST=/home/agent/workspaces/eurisko-study/lir-agent/experiments/E3/runs
mkdir -p $TMPR $DST
if [[ $KIND == eval ]]; then
  vm_ssh "$GPU_VM_IP" "sudo DEADLINE_MIN=${DEADLINE_MIN:-100} timeout --kill-after=30 104m bash /root/soup/E3/e3_remote_eval.sh $S $*" > $TMPR/remote-eval-s$S$(echo $* | tr -d ' ').log 2>&1
else
  vm_ssh "$GPU_VM_IP" "sudo DEADLINE_MIN=${DEADLINE_MIN:-104} timeout --kill-after=30 108m bash /root/soup/E3/e3_remote.sh $KIND $S $*" > $TMPR/remote-$KIND-s$S.log 2>&1
fi
rc=$?
if [[ $KIND == rolestrain ]]; then
  mkdir -p /tmp/lir-agent/e3/ck/s$S/roles
  rsync -a --rsync-path='sudo rsync' -e "ssh ${SSH_OPTS[*]}" "$SSH_USER@$GPU_VM_IP:/root/ck/" /tmp/lir-agent/e3/ck/s$S/roles/ || echo "ck sync failed" >&2
fi
rsync -az --rsync-path='sudo rsync' -e "ssh ${SSH_OPTS[*]}" --exclude='*.pt' "$SSH_USER@$GPU_VM_IP:/root/runs/E3/" "$TMPR/" || echo "sync failed" >&2
rsync -a --include='*/' --include='*.json' --include='*.jsonl' --include='*.csv' --include='*.png' --include='*.txt' --exclude='*' "$TMPR/" "$DST/"
echo "remote rc=$rc"
exit 0
