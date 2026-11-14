#!/bin/bash
# Launch one wm2 rental: launch.sh RUN_ID REMOTE_MINUTES JOBFILE [PAR]
#   JOBFILE: one `job.py` argument list per line (PAR at a time on the VM's one L4; default 6).
# Freezes the code into /tmp/wm2/src-RUN-EPOCH (harness env/ copied unmodified, round-1 smallnet code unmodified,
# wm2 code, T1/T2 data caches, remote.sh, the job file) and hands it to logical-soup/infra/gpu-run.sh
# (budget group wm2, cap $100). Refuses if wm2 spend incl. open reservations + this reserve > $99 (was $95 until 19:16Z; the guard still kills at $95 of actual spend), or if the run timeout would exceed 235 min.
# Never edit this file while launches run (round-1 lesson): copy, edit, rename.
set -u
RUN=$1; MIN=$2; JOBFILE=$3; PAR=${4:-6}
W=/home/agent/workspaces/eurisko-study/li-net/wm2
S=/home/agent/workspaces/eurisko-study/li-net/smallnet
H=/home/agent/workspaces/eurisko-study/lir-agent/harness
INF=/home/agent/workspaces/eurisko-study/logical-soup/infra
T=/tmp/wm2
mkdir -p $T/logs
L=$T/logs/launch-$RUN.log
[[ $RUN == wm2-* ]] || { echo "run id must start with wm2-" >&2; exit 2; }
[[ -f $JOBFILE ]] || { echo "no job file $JOBFILE" >&2; exit 2; }
source $INF/lib.sh
TO=$(( MIN + 12 ))
(( TO <= 235 )) || { echo "timeout ${TO}m > 235m" >&2; exit 2; }
RATE=0.87
reserve=$(awk -v m=$TO -v r=$RATE 'BEGIN{printf "%.2f", r*(m/60+0.6)}')
spent=$(spend_usd wm2)
if awk -v s=$spent -v r=$reserve 'BEGIN{exit !(s+r>99)}'; then
  echo "$(date -u +%FT%TZ) REFUSE $RUN: wm2 spent+reserved $spent + reserve $reserve > 99" | tee -a $L; exit 3
fi
SRC=$T/src-$RUN-$(date +%s)
mkdir -p $SRC/harness $SRC/smallnet $SRC/wm2
rsync -a --exclude=__pycache__ $H/env $SRC/harness/
cp $S/data.py $S/bninfer.py $S/netdef.py $S/smallnet.py $SRC/smallnet/
cp $W/wm2.py $W/job.py $W/t2data.py $SRC/wm2/
cp -r $T/data-t1 $T/data-t2 $SRC/
cp $W/gpu/remote.sh $W/gpu/remote_sched.py $SRC/
cp $JOBFILE $SRC/jobs.txt
lsha=$(sha256sum $W/LOOP.md 2>/dev/null | cut -c1-16)
echo "$(date -u +%FT%TZ) LAUNCH $RUN minutes=$MIN timeout=${TO}m par=$PAR spent=$spent reserve=$reserve jobs=$(grep -c -- '--task' $JOBFILE) loop_sha=$lsha src=$SRC" | tee -a $L
PROV=${WM2_PROV:-STANDARD}
echo "$(date -u +%FT%TZ) PROVISIONING $RUN $PROV" >> $L
GPU_STARTUP=minimal GPU_PROVISIONING=$PROV GPU_SOURCE_DIR=$SRC GPU_BUDGET_GROUP=wm2 GPU_GROUP_BUDGET_USD=100 GPU_RUN_TIMEOUT=${TO}m \
  $INF/gpu-run.sh --up --run-id $RUN --driver $W/gpu/driver.sh -- $MIN jobs.txt $PAR >> $L 2>&1
rc=$?
echo "$(date -u +%FT%TZ) END $RUN rc=$rc" | tee -a $L
exit $rc
