#!/bin/bash
# Round-3 rental: launch3.sh RUN_ID REMOTE_MINUTES JOBFILE [PAR]   (RUN_ID must start with eca3-)
# Like launch.sh, but ships ecadata3.py and the round-3 data (/tmp/eca/data3), and refuses if round-3 spend + this VM's
# reserve would exceed $40 (VMs named soup-gpu-eca3*) or the overnight-ca group would exceed $105.
set -u
RUN=$1; MIN=$2; JOBFILE=$3; PAR=${4:-2}
E=/home/agent/workspaces/eurisko-study/li-net/eca
S=/home/agent/workspaces/eurisko-study/li-net/smallnet
INF=/home/agent/workspaces/eurisko-study/logical-soup/infra
T=/tmp/eca
mkdir -p $T/logs
L=$T/logs/launch-$RUN.log
[[ $RUN == eca3-* ]] || { echo "run id must start with eca3-" >&2; exit 2; }
[[ -f $JOBFILE ]] || { echo "no job file $JOBFILE" >&2; exit 2; }
source $INF/lib.sh
TO=$(( MIN + 12 ))
(( TO <= 235 )) || { echo "timeout ${TO}m > 235m" >&2; exit 2; }
RATE=0.87
reserve=$(awk -v m=$TO -v r=$RATE 'BEGIN{printf "%.2f", r*(m/60+0.6)}')
spent=$(spend_usd overnight-ca)
r3=$(python3 - "$GPU_LOG" <<'PY'
import re, sys
res, cost = {}, {}
for l in open(sys.argv[1]):
    m = re.search(r"vm=(soup-gpu-eca3\S*)", l)
    if not m:
        continue
    n = m.group(1)
    r = re.search(r"reserve_usd=([\d.]+)", l)
    c = re.search(r"cost_usd=([\d.]+)", l)
    if (" PENDING-CREATE " in l or " CREATE vm=" in l) and r:
        res[n] = float(r.group(1))
    if (" DELETE vm=" in l or " COST-AUDIT vm=" in l) and c:
        cost[n] = float(c.group(1))
print(f"{sum(cost.get(n, v) for n, v in res.items()):.2f}")
PY
)
if awk -v s=$spent -v r=$reserve 'BEGIN{exit !(s+r>105)}'; then
  echo "$(date -u +%FT%TZ) REFUSE $RUN: group spent $spent + reserve $reserve > 105" | tee -a $L; exit 3
fi
if awk -v s=$r3 -v r=$reserve 'BEGIN{exit !(s+r>40)}'; then
  echo "$(date -u +%FT%TZ) REFUSE $RUN: round-3 spent/reserved $r3 + reserve $reserve > 40" | tee -a $L; exit 3
fi
SRC=$T/src-$RUN-$(date +%s)
mkdir -p $SRC/eca $SRC/smallnet $SRC/data
cp $E/ecadata.py $E/ecadata3.py $E/ecanet.py $E/ecarun.py $SRC/eca/
cp $S/netdef.py $SRC/smallnet/
cp $T/data3/*.npy $T/data3/*.npz $SRC/data/
cp $E/gpu/remote.sh $SRC/
cp $JOBFILE $SRC/jobs.txt
echo "$(date -u +%FT%TZ) LAUNCH $RUN minutes=$MIN timeout=${TO}m par=$PAR group_spent=$spent r3=$r3 reserve=$reserve jobs=$(grep -c -- '--rule' $JOBFILE) src=$SRC" | tee -a $L
PROV=${ECA_PROV:-STANDARD}
ZONES_ECA=${ECA_ZONES:-"us-central1-a us-central1-b us-central1-c us-east4-a us-east4-c us-east1-b us-east1-c us-east1-d us-west1-a us-west1-b us-west1-c us-west4-a us-west4-c"}
GPU_ZONES="$ZONES_ECA" GPU_STARTUP=minimal GPU_PROVISIONING=$PROV GPU_SOURCE_DIR=$SRC GPU_BUDGET_GROUP=overnight-ca GPU_GROUP_BUDGET_USD=110 GPU_RUN_TIMEOUT=${TO}m \
  $E/gpu/gpu-run-zones.sh --up --run-id $RUN --driver $E/gpu/driver.sh -- $MIN jobs.txt $PAR >> $L 2>&1
rc=$?
echo "$(date -u +%FT%TZ) END $RUN rc=$rc" | tee -a $L
exit $rc
