#!/bin/bash
# Launch one swarm rental: launch.sh RUN_ID REMOTE_MINUTES JOBFILE
#   JOBFILE: one run_swarm.py argument list per line (all lines run in parallel on the VM's one L4).
# Refuses unless spend_usd(swarm-predictor) + this VM's reserve <= $38 (coordinator rule) and fewer than 2 swarm
# VMs are live. Freezes the code into a fresh /tmp/swarm-predictor/src-RUN-EPOCH (harness env/ + model/ copied
# unmodified, swarm code, remote.sh, the job file) and hands it to gpu-run.sh.
set -u
RUN=$1; MIN=$2; JOBFILE=$3
S=/home/agent/workspaces/eurisko-study/li-net/swarm-predictor
H=/home/agent/workspaces/eurisko-study/lir-agent/harness
INF=/home/agent/workspaces/eurisko-study/logical-soup/infra
T=/tmp/swarm-predictor
mkdir -p $T/logs
L=$T/logs/launch-$RUN.log
[[ $RUN == swarm-* ]] || { echo "run id must start with swarm-" >&2; exit 2; }
[[ -f $JOBFILE ]] || { echo "no job file $JOBFILE" >&2; exit 2; }
source $INF/lib.sh
TO=$(( MIN + 15 ))
reserve=$(awk -v m=$TO 'BEGIN{printf "%.2f", 0.87*(m/60+0.6)}')
spent=$(spend_usd swarm-predictor)
live=$("${GC[@]}" compute instances list --filter='labels.logical-soup=1 AND name~soup-gpu-swarm' --format='value(name)' 2>/dev/null | grep -c .)
if awk -v s=$spent -v r=$reserve 'BEGIN{exit !(s+r>38)}'; then
  echo "$(date -u +%FT%TZ) REFUSE $RUN: spent $spent + reserve $reserve > 38" | tee -a $L; exit 3
fi
if (( live >= 2 )); then echo "$(date -u +%FT%TZ) REFUSE $RUN: $live swarm VMs live" | tee -a $L; exit 4; fi
SRC=$T/src-$RUN-$(date +%s)
mkdir -p $SRC/harness $SRC/swarm
rsync -a --exclude=__pycache__ $H/env $H/model $SRC/harness/
cp $S/pool.py $S/settle.py $S/refocus.py $S/run_swarm.py $SRC/swarm/
cp $S/gpu/remote.sh $SRC/
cp $JOBFILE $SRC/jobs.txt
echo "$(date -u +%FT%TZ) LAUNCH $RUN minutes=$MIN timeout=${TO}m spent=$spent reserve=$reserve live=$live src=$SRC" | tee -a $L
sed 's/^/    job: /' $JOBFILE | tee -a $L
GPU_SOURCE_DIR=$SRC GPU_BUDGET_GROUP=swarm-predictor GPU_GROUP_BUDGET_USD=40 GPU_RUN_TIMEOUT=${TO}m \
  $INF/gpu-run.sh --up --run-id $RUN --driver $S/gpu/driver.sh -- $MIN jobs.txt >> $L 2>&1
echo "$(date -u +%FT%TZ) END $RUN rc=$?" | tee -a $L
