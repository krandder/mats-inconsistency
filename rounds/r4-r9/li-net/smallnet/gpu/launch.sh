#!/bin/bash
# Launch one smallnet rental: launch.sh RUN_ID REMOTE_MINUTES JOBFILE [PAR] [--a100]
#   JOBFILE: one smallnet.py argument list per line (run PAR at a time on the VM's one GPU; default 6).
# Freezes the code into /tmp/smallnet/src-RUN-EPOCH (harness env/ copied unmodified, smallnet code, data cache,
# remote.sh, the job file) and hands it to logical-soup/infra/gpu-run.sh (budget group smallnet, cap $200).
# Refuses if smallnet spend + reserve > $190, or if the run timeout would exceed 235 min.
set -u
RUN=$1; MIN=$2; JOBFILE=$3; PAR=${4:-6}; A100=${5:-}
S=/home/agent/workspaces/eurisko-study/li-net/smallnet
H=/home/agent/workspaces/eurisko-study/lir-agent/harness
INF=/home/agent/workspaces/eurisko-study/logical-soup/infra
T=/tmp/smallnet
mkdir -p $T/logs
L=$T/logs/launch-$RUN.log
[[ $RUN == smallnet-* ]] || { echo "run id must start with smallnet-" >&2; exit 2; }
[[ -f $JOBFILE ]] || { echo "no job file $JOBFILE" >&2; exit 2; }
source $INF/lib.sh
TO=$(( MIN + 12 ))
(( TO <= 235 )) || { echo "timeout ${TO}m > 235m" >&2; exit 2; }
RATE=0.87; [[ $A100 == --a100 ]] && RATE=3.69
reserve=$(awk -v m=$TO -v r=$RATE 'BEGIN{printf "%.2f", r*(m/60+0.6)}')
spent=$(spend_usd smallnet)
if awk -v s=$spent -v r=$reserve 'BEGIN{exit !(s+r>190)}'; then
  echo "$(date -u +%FT%TZ) REFUSE $RUN: spent $spent + reserve $reserve > 190" | tee -a $L; exit 3
fi
SRC=$T/src-$RUN-$(date +%s)
mkdir -p $SRC/harness $SRC/smallnet $SRC/data
rsync -a --exclude=__pycache__ $H/env $SRC/harness/
cp $S/data.py $S/bninfer.py $S/netdef.py $S/smallnet.py $SRC/smallnet/
cp $T/data/*.npz $T/data/*.npy $SRC/data/
cp $S/gpu/remote.sh $SRC/
cp $JOBFILE $SRC/jobs.txt
echo "$(date -u +%FT%TZ) LAUNCH $RUN minutes=$MIN timeout=${TO}m par=$PAR spent=$spent reserve=$reserve jobs=$(grep -c -- '--arm' $JOBFILE) src=$SRC" | tee -a $L
UPA=(); [[ $A100 == --a100 ]] && UPA=(--a100)
# 2026-10-10 17:20Z: on-demand (STANDARD) by default after 10 of 14 spot L4s were preempted within 35 min; SMALLNET_PROV=SPOT to override
PROV=${SMALLNET_PROV:-STANDARD}
echo "$(date -u +%FT%TZ) PROVISIONING $RUN $PROV" >> $L
GPU_PROVISIONING=$PROV GPU_SOURCE_DIR=$SRC GPU_BUDGET_GROUP=smallnet GPU_GROUP_BUDGET_USD=200 GPU_RUN_TIMEOUT=${TO}m \
  $INF/gpu-run.sh --up "${UPA[@]}" --run-id $RUN --driver $S/gpu/driver.sh -- $MIN jobs.txt $PAR >> $L 2>&1
rc=$?
echo "$(date -u +%FT%TZ) END $RUN rc=$rc" | tee -a $L
exit $rc
