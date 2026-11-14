#!/bin/bash
# launch.sh RUN_ID REMOTE_MINUTES JOBFILE PAR   (RUN_ID must start with la-). Budget group linear-agg, cap $10.
# Freezes the linear-agg source into /tmp/linear-agg/src-RUN-*, then runs eca's zone-flexible copy of gpu-run.sh (unchanged)
# with this directory's driver; the VM is deleted by gpu-run on exit.
set -u
RUN=$1; MIN=$2; JOBFILE=$3; PAR=${4:-2}
LA=/home/agent/workspaces/mats/work/linear-agg
E=/home/agent/workspaces/eurisko-study/li-net/eca
[[ $RUN == la-* ]] || { echo "run id must start with la-" >&2; exit 2; }
[[ -f $JOBFILE ]] || { echo "no job file $JOBFILE" >&2; exit 2; }
source /home/agent/workspaces/eurisko-study/logical-soup/infra/lib.sh
TO=$(( MIN + 10 ))
SRC=/tmp/linear-agg/src-$RUN-$(date +%s)
mkdir -p $SRC/gpu
cp $LA/synth.py $LA/lt.py $LA/train_a.py $LA/b_rollouts.py $SRC/
cp -r $LA/eca $LA/ckpt $LA/data3 $SRC/
rm -rf $SRC/eca/__pycache__
cp $LA/gpu/remote.sh $SRC/gpu/
cp $JOBFILE $SRC/gpu/jobs.txt
mkdir -p $LA/gpu/logs
L=$LA/gpu/logs/launch-$RUN.log
echo "$(date -u +%FT%TZ) LAUNCH $RUN minutes=$MIN timeout=${TO}m par=$PAR group_spent=$(spend_usd linear-agg) jobs=$(grep -vc '^#' $JOBFILE) src=$SRC" | tee -a $L
PROV=${LA_PROV:-SPOT}
ZONES=${LA_ZONES:-"us-central1-a us-central1-b us-central1-c us-east4-a us-east4-c us-east1-b us-east1-c us-east1-d us-west1-a us-west1-b us-west4-a"}
GPU_ZONES="$ZONES" GPU_STARTUP=minimal GPU_PROVISIONING=$PROV GPU_SOURCE_DIR=$SRC GPU_BUDGET_GROUP=linear-agg GPU_GROUP_BUDGET_USD=10 GPU_RUN_TIMEOUT=${TO}m \
  $E/gpu/gpu-run-zones.sh --up --run-id $RUN --driver $LA/gpu/driver.sh -- $MIN /root/soup/gpu/jobs.txt $PAR >> $L 2>&1
rc=$?
echo "$(date -u +%FT%TZ) END $RUN rc=$rc" | tee -a $L
exit $rc
