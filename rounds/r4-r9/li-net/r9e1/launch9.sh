#!/bin/bash
# li-net/r9e1/launch9.sh RUN_ID TAG "SEEDS" REMOTE_MIN [extra e1v6.py flags]
# One L4 VM through logical-soup/infra/gpu-run.sh only (budget group r9, group cap $1.35 shared with R9-E2; R9-E1 own
# ceiling $0.60 is checked by ledger9.py before every launch). gpu-run.sh deletes the VM when the driver returns (trap
# cleanup EXIT), i.e. as soon as the registered run and the final sync are done. Retries only when the VM create failed.
set -u
RUN=$1; TAG=$2; SEEDS=$3; RMIN=$4; shift 4
R9=/home/agent/workspaces/eurisko-study/li-net/r9e1
INFRA=/home/agent/workspaces/eurisko-study/logical-soup/infra
mkdir -p /tmp/r9e1/logs
LOG=/tmp/r9e1/logs/launch-$RUN.log
[[ $RUN == r9e1-* ]] || { echo "run id must start with r9e1-" >&2; exit 2; }
bash $R9/stage9.sh || exit 2
python3 $R9/ledger9.py --check-launch "$(( RMIN + 3 ))" || { echo "$(date -u +%FT%TZ) REFUSE $RUN (R9-E1 ceiling)" >> $LOG; exit 3; }
export GPU_SOURCE_DIR=/tmp/r9e1/src GPU_BUDGET_GROUP=r9 GPU_GROUP_BUDGET_USD=1.35 GPU_STARTUP=minimal
export GPU_RUN_TIMEOUT=$(( RMIN + 3 ))m   # gpu-run.sh timeout = remote + 3 min (as R8 after its deviation 4)
for i in 1 2 3 4 5 6 7 8; do
  echo "$(date -u +%FT%TZ) LAUNCH $RUN try $i tag=$TAG seeds=$SEEDS remote=${RMIN}m timeout=$GPU_RUN_TIMEOUT prov=${GPU_PROVISIONING:-SPOT} args=$*" >> $LOG
  $INFRA/gpu-run.sh --up --run-id $RUN --driver $R9/driver9.sh -- $TAG "$SEEDS" $RMIN "$@" >> $LOG 2>&1
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
