#!/bin/bash
# li-net/r7e1/launch7.sh RUN_ID TAG "SEEDS" REMOTE_MIN [extra e1v4.py flags]
# One L4 VM through logical-soup/infra/gpu-run.sh only (budget group r7, group cap $6.50 shared with R7-E3; R7-E1's own
# ceiling $2.50 is checked by ledger7.py before every launch). Retries only when the VM create failed (stockout in all zones).
# STAGE_CKPT=1 (recovery only) stages the synced theta_real checkpoints so the rerun loads them instead of retraining.
set -u
RUN=$1; TAG=$2; SEEDS=$3; RMIN=$4; shift 4
R7=/home/agent/workspaces/eurisko-study/li-net/r7e1
INFRA=/home/agent/workspaces/eurisko-study/logical-soup/infra
mkdir -p /tmp/r7e1/logs
LOG=/tmp/r7e1/logs/launch-$RUN.log
[[ $RUN == r7e1-* ]] || { echo "run id must start with r7e1-" >&2; exit 2; }
bash $R7/stage7.sh || exit 2
python3 $R7/ledger7.py --check-launch "$(( RMIN + 10 ))" || { echo "$(date -u +%FT%TZ) REFUSE $RUN (R7-E1 ceiling)" >> $LOG; exit 3; }
export GPU_SOURCE_DIR=/tmp/r7e1/src GPU_BUDGET_GROUP=r7 GPU_GROUP_BUDGET_USD=6.5 GPU_STARTUP=minimal
export GPU_RUN_TIMEOUT=$(( RMIN + 10 ))m
for i in 1 2 3 4 5 6 7 8; do
  echo "$(date -u +%FT%TZ) LAUNCH $RUN try $i tag=$TAG seeds=$SEEDS remote=${RMIN}m timeout=$GPU_RUN_TIMEOUT prov=${GPU_PROVISIONING:-SPOT} stage_ckpt=${STAGE_CKPT:-0} args=$*" >> $LOG
  $INFRA/gpu-run.sh --up --run-id $RUN --driver $R7/driver7.sh -- $TAG "$SEEDS" $RMIN "$@" >> $LOG 2>&1
  rc=$?
  att=$(awk -v r="LAUNCH $RUN try $i " 'index($0, r){buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' $LOG)
  if [[ $rc != 0 ]] && ! grep -q "READY:" <<<"$att" && grep -qiE "resources|stockout|exhausted|CREATE-FAILED|gcloud.compute.instances.create" <<<"$att"; then
    echo "$(date -u +%FT%TZ) RETRY $RUN after create failure" >> $LOG
    sleep 60
    continue
  fi
  echo "$(date -u +%FT%TZ) END $RUN rc=$rc" >> $LOG
  exit $rc
done
echo "$(date -u +%FT%TZ) GIVEUP $RUN" >> $LOG
exit 9
