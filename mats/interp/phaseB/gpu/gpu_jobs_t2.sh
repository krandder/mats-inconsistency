#!/usr/bin/env bash
# E40 job on one L4 VM (PREREG-B amendment 48): as gpu_jobs.sh, but each job-file line is NAME followed by train_t2.py arguments.
# Up to 8 trainings at once; no new training after DL_START seconds, everything stopped at DL_STOP seconds.
# usage: bash gpu_jobs_t2.sh RID JOBFILE [DL_START DL_STOP]
set -x
RID=$1; JOBS=$2; DL_START=${3:-15600}; DL_STOP=${4:-17400}; PY=/root/venv/bin/python; OUT=/root/runs/$RID
nvidia-smi -L
mkdir -p $OUT
cp $JOBS $OUT/jobs.txt
export OMP_NUM_THREADS=1
T0=$(date +%s)
jobs_running() { jobs -rp | wc -l; }
while read -r NAME ARGS; do
  [ -z "$NAME" ] && continue
  while [ "$(jobs_running)" -ge 8 ]; do sleep 20; done
  if [ $(( $(date +%s) - T0 )) -ge $DL_START ]; then echo "deadline: not starting $NAME"; continue; fi
  $PY train_t2.py $ARGS --out $OUT/$NAME < /dev/null > $OUT/$NAME.log 2>&1 &
  sleep 3
done < $JOBS
while [ "$(jobs_running)" -gt 0 ]; do
  if [ $(( $(date +%s) - T0 )) -ge $DL_STOP ]; then echo "hard stop: killing $(jobs_running) trainings"; kill $(jobs -rp); break; fi
  sleep 30
done
wait
ls $OUT
echo JOBS-DONE
