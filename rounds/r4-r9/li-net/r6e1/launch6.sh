#!/bin/bash
# li-net/r6e1/launch6.sh RUN_ID TAG "SEEDS" REMOTE_MIN [extra e1v3.py flags]
# One L4 VM through logical-soup/infra/gpu-run.sh only (budget group r6, group cap $12 shared with R6-E3; R6-E1's own ceiling
# $2.50 is checked by ledger6.py before every launch). Retries only when the VM create failed (stockout in all zones).
set -u
RUN=$1; TAG=$2; SEEDS=$3; RMIN=$4; shift 4
R6=/home/agent/workspaces/eurisko-study/li-net/r6e1
INFRA=/home/agent/workspaces/eurisko-study/logical-soup/infra
mkdir -p /tmp/r6e1/logs
LOG=/tmp/r6e1/logs/launch-$RUN.log
[[ $RUN == r6e1-* ]] || { echo "run id must start with r6e1-" >&2; exit 2; }
bash $R6/stage6.sh || exit 2
python3 $R6/ledger6.py --check-launch "$(( RMIN + 10 ))" || { echo "$(date -u +%FT%TZ) REFUSE $RUN (R6-E1 ceiling)" >> $LOG; exit 3; }
export GPU_SOURCE_DIR=/tmp/r6e1/src GPU_BUDGET_GROUP=r6 GPU_GROUP_BUDGET_USD=12 GPU_STARTUP=minimal
export GPU_RUN_TIMEOUT=$(( RMIN + 10 ))m
for i in 1 2 3 4 5 6 7 8; do
  echo "$(date -u +%FT%TZ) LAUNCH $RUN try $i tag=$TAG seeds=$SEEDS remote=${RMIN}m timeout=$GPU_RUN_TIMEOUT prov=${GPU_PROVISIONING:-SPOT} args=$*" >> $LOG
  $INFRA/gpu-run.sh --up --run-id $RUN --driver $R6/driver6.sh -- $TAG "$SEEDS" $RMIN "$@" >> $LOG 2>&1
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
