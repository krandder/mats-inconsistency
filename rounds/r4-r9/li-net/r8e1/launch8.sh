#!/bin/bash
# li-net/r8e1/launch8.sh RUN_ID TAG "SEEDS" REMOTE_MIN [extra e1v5.py flags]
# One L4 VM through logical-soup/infra/gpu-run.sh only (budget group r8, group cap $2.20 shared with R8-E2 and R8-E3; R8-E1's
# own ceiling $0.80 is checked by ledger8.py before every launch). gpu-run.sh deletes the VM when the driver returns (trap
# cleanup EXIT), i.e. as soon as the registered run and the final sync are done. Retries only when the VM create failed.
set -u
RUN=$1; TAG=$2; SEEDS=$3; RMIN=$4; shift 4
R8=/home/agent/workspaces/eurisko-study/li-net/r8e1
INFRA=/home/agent/workspaces/eurisko-study/logical-soup/infra
mkdir -p /tmp/r8e1/logs
LOG=/tmp/r8e1/logs/launch-$RUN.log
[[ $RUN == r8e1-* ]] || { echo "run id must start with r8e1-" >&2; exit 2; }
bash $R8/stage8.sh || exit 2
python3 $R8/ledger8.py --check-launch "$(( RMIN + 3 ))" || { echo "$(date -u +%FT%TZ) REFUSE $RUN (R8-E1 ceiling)" >> $LOG; exit 3; }
export GPU_SOURCE_DIR=/tmp/r8e1/src GPU_BUDGET_GROUP=r8 GPU_GROUP_BUDGET_USD=2.2 GPU_STARTUP=minimal
export GPU_RUN_TIMEOUT=$(( RMIN + 3 ))m   # PREREG deviation 4 (was RMIN + 10)
for i in 1 2 3 4 5 6 7 8; do
  echo "$(date -u +%FT%TZ) LAUNCH $RUN try $i tag=$TAG seeds=$SEEDS remote=${RMIN}m timeout=$GPU_RUN_TIMEOUT prov=${GPU_PROVISIONING:-SPOT} args=$*" >> $LOG
  $INFRA/gpu-run.sh --up --run-id $RUN --driver $R8/driver8.sh -- $TAG "$SEEDS" $RMIN "$@" >> $LOG 2>&1
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
