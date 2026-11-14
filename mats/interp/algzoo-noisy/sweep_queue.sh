#!/bin/bash
# Amendment 8 dispatcher: runs the job lines of sweep_jobs.txt in order, at most 4 heavy python processes at a time.
# Line format: "WAITFOR_FILE|COMMAND" (WAITFOR_FILE may be "-").
cd /home/agent/workspaces/mats/work/algzoo-noisy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 WARM_STEPS=1024
PY=/home/agent/.venvs/aggext/bin/python
count() { pgrep -fc "^/home/agent/.venvs/aggext/bin/python (train_batched|sweep_an|zoo_prep)\.py"; }
while IFS='|' read -r wf cmd; do
  [ -z "$cmd" ] && continue
  if [ "$wf" != "-" ]; then until [ -f "$wf" ]; do sleep 30; done; fi
  until [ "$(count)" -lt 4 ]; do sleep 20; done
  echo "$(date -u +%H:%M:%S) START $cmd" >> logs/sweep/queue.log
  nohup nice -n 19 ionice -c3 bash -c "$cmd" >> logs/sweep/queue_jobs.log 2>&1 &
  sleep 5
done < sweep_jobs.txt
echo "$(date -u +%H:%M:%S) ALL DISPATCHED" >> logs/sweep/queue.log
