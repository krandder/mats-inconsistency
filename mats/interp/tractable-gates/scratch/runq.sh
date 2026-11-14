#!/usr/bin/env bash
# Queue runner: run the commands in QUEUEFILE (one per line: NAME then the command, run from the lane dir) with at most P at
# once, each at nice 19 / ionice -c3 with one BLAS/OpenMP thread. Logs: logs/q_NAME.log; completion lines in logs/queue.done.
# usage: bash scratch/runq.sh QUEUEFILE P
cd /home/agent/workspaces/mats/work/tractable-gates
QF=$1; P=${2:-2}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NTHREADS=1
PY=/home/agent/.venvs/arc-noisy/bin/python
while read -r NAME CMD; do
  [ -z "$NAME" ] && continue
  case "$NAME" in \#*) continue;; esac
  while [ "$(jobs -rp | wc -l)" -ge "$P" ]; do sleep 5; done
  ( nice -n 19 ionice -c3 $PY $CMD > logs/q_$NAME.log 2>&1; echo "$(date -u +%H:%M:%SZ) $NAME exit=$?" >> logs/queue.done ) &
  sleep 1
done < "$QF"
wait
echo "$(date -u +%H:%M:%SZ) QUEUE $QF ALL-DONE" >> logs/queue.done
