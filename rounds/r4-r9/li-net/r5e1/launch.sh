#!/bin/bash
# li-net/r5e1/launch.sh RUN_ID TAG "SEEDS" REMOTE_MIN [extra run.py flags]
# One L4 VM through logical-soup/infra/gpu-run.sh only (budget group r5, group cap $22 shared with R5-E2; R5-E1's own
# ceiling $6 is checked by ledger.py before every launch). Retries only when the VM create failed (stockout in all zones).
set -u
RUN=$1; TAG=$2; SEEDS=$3; RMIN=$4; shift 4
R5=/home/agent/workspaces/eurisko-study/li-net/r5e1
INFRA=/home/agent/workspaces/eurisko-study/logical-soup/infra
mkdir -p /tmp/r5e1/logs
LOG=/tmp/r5e1/logs/launch-$RUN.log
[[ $RUN == r5e1-* ]] || { echo "run id must start with r5e1-" >&2; exit 2; }
bash $R5/stage.sh || exit 2
python3 $R5/ledger.py --check-launch "$(( RMIN + 10 ))" || { echo "$(date -u +%FT%TZ) REFUSE $RUN (R5-E1 ceiling)" >> $LOG; exit 3; }
export GPU_SOURCE_DIR=/tmp/r5e1/src GPU_BUDGET_GROUP=r5 GPU_GROUP_BUDGET_USD=22 GPU_STARTUP=minimal
export GPU_RUN_TIMEOUT=$(( RMIN + 10 ))m
for i in 1 2 3 4 5 6 7 8; do
  echo "$(date -u +%FT%TZ) LAUNCH $RUN try $i tag=$TAG seeds=$SEEDS remote=${RMIN}m timeout=$GPU_RUN_TIMEOUT prov=${GPU_PROVISIONING:-SPOT} args=$*" >> $LOG
  $INFRA/gpu-run.sh --up --run-id $RUN --driver $R5/driver.sh -- $TAG "$SEEDS" $RMIN "$@" >> $LOG 2>&1
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
