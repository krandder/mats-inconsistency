#!/usr/bin/env bash
# Generic GPU job (fourth agent, from E43 on): run the trainings listed in a job file (one per line: NAME, then arguments of
# SCRIPT), in file order, up to 8 at once. Own deadline: no new training after DL_START seconds, everything stopped at DL_STOP
# seconds. Outputs: /root/runs/$RID/<NAME>/ and <NAME>.log.
# usage: bash gpu_jobs_x.sh RID JOBFILE SCRIPT [DL_START DL_STOP]
set -x
RID=$1; JOBS=$2; SCRIPT=$3; DL_START=${4:-15600}; DL_STOP=${5:-17400}; PY=/root/venv/bin/python; OUT=/root/runs/$RID
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
  $PY $SCRIPT $ARGS --out $OUT/$NAME < /dev/null > $OUT/$NAME.log 2>&1 &
  sleep 3
done < $JOBS
while [ "$(jobs_running)" -gt 0 ]; do
  if [ $(( $(date +%s) - T0 )) -ge $DL_STOP ]; then echo "hard stop: killing $(jobs_running) trainings"; kill $(jobs -rp); break; fi
  sleep 30
done
wait
ls $OUT
echo JOBS-DONE
