#!/bin/bash
# Local driver for logical-soup/infra/gpu-run.sh --driver. Args: TAG "SEEDS" TIMEOUT_MIN [extra e1v5.py flags]
# Syncs back ONLY *.json / *.csv / *.png of the run -> li-net/r8e1/results/TAG/ (every 4 min and at the end); text logs ->
# /tmp/r8e1/logs/TAG/. Nothing else leaves the VM; gpu-run.sh deletes the VM as soon as this driver returns (trap on EXIT).
set -u
TAG=$1; SEEDS=$2; TMIN=$3; shift 3
R8=/home/agent/workspaces/eurisko-study/li-net/r8e1
DEST=$R8/results/$TAG; LOGS=/tmp/r8e1/logs/$TAG; mkdir -p $DEST $LOGS
source /home/agent/workspaces/eurisko-study/logical-soup/infra/lib.sh
RSH="ssh ${SSH_OPTS[*]}"; R="$SSH_USER@$GPU_VM_IP"
pull() {
  rsync -az --rsync-path='sudo rsync' -e "$RSH" --include='*/' --include='*.json' --include='*.csv' --include='*.png' \
    --exclude='*' "$R:/root/runs/r8e1/$TAG/" $DEST/ 2>/dev/null
  rsync -az --rsync-path='sudo rsync' -e "$RSH" --include='*/' --include='*.txt' --exclude='*' \
    "$R:/root/runs/r8e1/$TAG/" $LOGS/ 2>/dev/null
}
( while sleep 240; do pull; done ) & SYNC=$!
vm_ssh "$GPU_VM_IP" "sudo timeout --kill-after=30 ${TMIN}m bash /root/soup/remote8.sh $TAG '$SEEDS' $*" >> $LOGS/driver.txt 2>&1
rc=$?
kill $SYNC 2>/dev/null
pull || pull || echo "final sync failed" >&2
echo "$(date -u +%FT%TZ) remote rc=$rc" >> $LOGS/driver.txt
exit 0
