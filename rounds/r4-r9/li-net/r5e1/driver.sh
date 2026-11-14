#!/bin/bash
# Local driver for logical-soup/infra/gpu-run.sh --driver. Args: TAG "SEEDS" TIMEOUT_MIN [extra run.py flags]
# Syncs ONLY *.json / *.csv / *.png back to li-net/r5e1/results/TAG/ (every 4 min and at the end); text logs go to
# /tmp/r5e1/logs/TAG/; nothing else (no checkpoints) leaves the VM, which gpu-run.sh deletes afterwards.
set -u
TAG=$1; SEEDS=$2; TMIN=$3; shift 3
R5=/home/agent/workspaces/eurisko-study/li-net/r5e1
DEST=$R5/results/$TAG; LOGS=/tmp/r5e1/logs/$TAG; mkdir -p $DEST $LOGS
source /home/agent/workspaces/eurisko-study/logical-soup/infra/lib.sh
RSH="ssh ${SSH_OPTS[*]}"; R="$SSH_USER@$GPU_VM_IP"
pull() {
  rsync -az --rsync-path='sudo rsync' -e "$RSH" --include='*/' --include='*.json' --include='*.csv' --include='*.png' \
    --exclude='*' "$R:/root/runs/r5e1/$TAG/" $DEST/ 2>/dev/null
  rsync -az --rsync-path='sudo rsync' -e "$RSH" --include='*/' --include='*.txt' --exclude='*' \
    "$R:/root/runs/r5e1/$TAG/" $LOGS/ 2>/dev/null
}
( while sleep 240; do pull; done ) & SYNC=$!
vm_ssh "$GPU_VM_IP" "sudo timeout --kill-after=30 ${TMIN}m bash /root/soup/remote.sh $TAG '$SEEDS' $*" >> $LOGS/driver.txt 2>&1
rc=$?
kill $SYNC 2>/dev/null
pull || pull || echo "final sync failed" >&2
echo "$(date -u +%FT%TZ) remote rc=$rc" >> $LOGS/driver.txt
exit 0
