#!/bin/bash
# Local driver for logical-soup/infra/gpu-run.sh --driver. Args: TAG "SEEDS" TIMEOUT_MIN [extra e1v4.py flags]
# Syncs back ONLY: *.json / *.csv / *.png of the run -> li-net/r7e1/results/TAG/; the theta_real checkpoints
# (s<seed>/theta_real_v2.npz + .json, the spec's checkpoint-caching rule) -> li-net/r7e1/ckpt/; text logs -> /tmp/r7e1/logs/TAG/.
# Every 4 min and at the end; nothing else leaves the VM, which gpu-run.sh deletes afterwards.
set -u
TAG=$1; SEEDS=$2; TMIN=$3; shift 3
R7=/home/agent/workspaces/eurisko-study/li-net/r7e1
DEST=$R7/results/$TAG; CKD=$R7/ckpt; LOGS=/tmp/r7e1/logs/$TAG; mkdir -p $DEST $CKD $LOGS
source /home/agent/workspaces/eurisko-study/logical-soup/infra/lib.sh
RSH="ssh ${SSH_OPTS[*]}"; R="$SSH_USER@$GPU_VM_IP"
pull() {
  rsync -az --rsync-path='sudo rsync' -e "$RSH" --include='*/' --include='*.json' --include='*.csv' --include='*.png' \
    --exclude='*' "$R:/root/runs/r7e1/$TAG/" $DEST/ 2>/dev/null
  rsync -az --rsync-path='sudo rsync' -e "$RSH" --include='s*/' --include='theta_real_v2.npz' --include='theta_real_v2.json' \
    --exclude='*' "$R:/root/runs/r7e1/ckpt/" $CKD/ 2>/dev/null
  rsync -az --rsync-path='sudo rsync' -e "$RSH" --include='*/' --include='*.txt' --exclude='*' \
    "$R:/root/runs/r7e1/$TAG/" $LOGS/ 2>/dev/null
}
( while sleep 240; do pull; done ) & SYNC=$!
vm_ssh "$GPU_VM_IP" "sudo timeout --kill-after=30 ${TMIN}m bash /root/soup/remote7.sh $TAG '$SEEDS' $*" >> $LOGS/driver.txt 2>&1
rc=$?
kill $SYNC 2>/dev/null
pull || pull || echo "final sync failed" >&2
echo "$(date -u +%FT%TZ) remote rc=$rc" >> $LOGS/driver.txt
exit 0
