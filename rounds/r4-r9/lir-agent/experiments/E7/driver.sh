#!/bin/bash
# Local driver for logical-soup/infra/gpu-run.sh --driver. Arg: MODE (w1a w1b w2a w2b w2c w3).
# Uploads what the mode needs, runs remote.sh under a 105-minute timeout, then syncs back ONLY
# json/jsonl/log/csv/png to experiments/E7/{runs,an}; checkpoints and eval pickles go to /tmp/lir-agent/e7.
set -u
MODE=$1
E=/home/agent/workspaces/eurisko-study/lir-agent/experiments/E7
T=/tmp/lir-agent/e7
source /home/agent/workspaces/eurisko-study/logical-soup/infra/lib.sh
RSH="ssh ${SSH_OPTS[*]}"
mkdir -p $E/runs $E/an $T/ckpt $T/evalrecs
case $MODE in
  w2c2) rsync -az --rsync-path='sudo rsync' -e "$RSH" --include='warm-B0-*/' --include='B0-s4/' --include='*.pt' --exclude='*' \
         $T/ckpt/ "$SSH_USER@$GPU_VM_IP:/root/runs/ckpt/"
       rsync -az --rsync-path='sudo rsync' -e "$RSH" --include='*.json' --exclude='*' $E/an/ "$SSH_USER@$GPU_VM_IP:/root/runs/an/" ;;
  w2*) rsync -az --rsync-path='sudo rsync' -e "$RSH" --include='warm-*/' --include='pre_rl.pt' --exclude='*' \
         $T/ckpt/ "$SSH_USER@$GPU_VM_IP:/root/runs/ckpt/"
       rsync -az --rsync-path='sudo rsync' -e "$RSH" --include='*.json' --exclude='*' $E/an/ "$SSH_USER@$GPU_VM_IP:/root/runs/an/" ;;
  w3*) rsync -azL --rsync-path='sudo rsync' -e "$RSH" $T/ck3/ "$SSH_USER@$GPU_VM_IP:/root/runs/ck3/"
       rsync -az --rsync-path='sudo rsync' -e "$RSH" --include='*.json' --exclude='*' $E/an/ "$SSH_USER@$GPU_VM_IP:/root/runs/an/" ;;
esac
vm_ssh "$GPU_VM_IP" "sudo timeout --kill-after=30 105m bash /root/soup/e7/remote.sh $MODE" > $T/driver-$MODE.log 2>&1
rc=$?
R="$SSH_USER@$GPU_VM_IP"
rsync -az --rsync-path='sudo rsync' -e "$RSH" --include='*/' --include='*.json' --include='*.jsonl' --include='*.log' \
  --exclude='*' "$R:/root/runs/E7/" $E/runs/ || echo "sync runs failed" >&2
rsync -az --rsync-path='sudo rsync' -e "$RSH" --include='*.json' --exclude='*' "$R:/root/runs/an/" $E/an/ || echo "sync an failed" >&2
rsync -az --rsync-path='sudo rsync' -e "$RSH" --include='*/' --include='*.pkl' --exclude='*' "$R:/root/runs/E7/" $T/evalrecs/ || true
rsync -az --rsync-path='sudo rsync' -e "$RSH" "$R:/root/runs/ckpt/" $T/ckpt/ || echo "sync ckpt failed" >&2
echo "remote rc=$rc"
exit 0
