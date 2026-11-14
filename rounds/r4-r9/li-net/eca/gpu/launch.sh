#!/bin/bash
# Launch one E-CA rental: launch.sh RUN_ID REMOTE_MINUTES JOBFILE [PAR]
#   JOBFILE: one ecarun.py argument list per line (PAR at a time on the VM's one L4; default 2).
# Freezes the code into /tmp/eca/src-RUN-EPOCH (eca/*.py, smallnet/netdef.py copied unmodified, data cache, remote.sh,
# the job file) and hands it to logical-soup/infra/gpu-run.sh (budget group overnight-ca, cap $110).
# Refuses if overnight-ca spend + reserve > $105, or if the run timeout would exceed 235 min.
set -u
RUN=$1; MIN=$2; JOBFILE=$3; PAR=${4:-2}
E=/home/agent/workspaces/eurisko-study/li-net/eca
S=/home/agent/workspaces/eurisko-study/li-net/smallnet
INF=/home/agent/workspaces/eurisko-study/logical-soup/infra
T=/tmp/eca
mkdir -p $T/logs
L=$T/logs/launch-$RUN.log
[[ $RUN == eca-* ]] || { echo "run id must start with eca-" >&2; exit 2; }
[[ -f $JOBFILE ]] || { echo "no job file $JOBFILE" >&2; exit 2; }
source $INF/lib.sh
TO=$(( MIN + 12 ))
(( TO <= 235 )) || { echo "timeout ${TO}m > 235m" >&2; exit 2; }
RATE=0.87
reserve=$(awk -v m=$TO -v r=$RATE 'BEGIN{printf "%.2f", r*(m/60+0.6)}')
spent=$(spend_usd overnight-ca)
if awk -v s=$spent -v r=$reserve 'BEGIN{exit !(s+r>105)}'; then
  echo "$(date -u +%FT%TZ) REFUSE $RUN: spent $spent + reserve $reserve > 105" | tee -a $L; exit 3
fi
SRC=$T/src-$RUN-$(date +%s)
mkdir -p $SRC/eca $SRC/smallnet $SRC/data
cp $E/ecadata.py $E/ecanet.py $E/ecarun.py $SRC/eca/
cp $S/netdef.py $SRC/smallnet/
cp $T/data/*.npy $T/data/*.npz $SRC/data/
cp $E/gpu/remote.sh $SRC/
cp $JOBFILE $SRC/jobs.txt
echo "$(date -u +%FT%TZ) LAUNCH $RUN minutes=$MIN timeout=${TO}m par=$PAR spent=$spent reserve=$reserve jobs=$(grep -c -- '--rule' $JOBFILE) src=$SRC" | tee -a $L
PROV=${ECA_PROV:-STANDARD}
echo "$(date -u +%FT%TZ) PROVISIONING $RUN $PROV" >> $L
# 2026-10-15 01:41Z: zone-flexible eca copy of gpu-run/gpu-up (on-demand L4 stockouts in all us-central1 zones)
ZONES_ECA=${ECA_ZONES:-"us-central1-a us-central1-b us-central1-c us-east4-a us-east4-c us-east1-b us-east1-c us-east1-d us-west1-a us-west1-b us-west1-c us-west4-a us-west4-c"}
GPU_ZONES="$ZONES_ECA" GPU_STARTUP=minimal GPU_PROVISIONING=$PROV GPU_SOURCE_DIR=$SRC GPU_BUDGET_GROUP=overnight-ca GPU_GROUP_BUDGET_USD=110 GPU_RUN_TIMEOUT=${TO}m \
  /home/agent/workspaces/eurisko-study/li-net/eca/gpu/gpu-run-zones.sh --up --run-id $RUN --driver $E/gpu/driver.sh -- $MIN jobs.txt $PAR >> $L 2>&1
rc=$?
echo "$(date -u +%FT%TZ) END $RUN rc=$rc" | tee -a $L
exit $rc
