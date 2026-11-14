#!/usr/bin/env bash
# E2 queue: 2L seeds 4, 2, 1 then 3L (each waits for a free slot, <= 4 active analysis processes)
cd /home/agent/workspaces/mats/work/cm-interp-B
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/agent/workspaces/mats/work/tiny-pfn
for r in L2_s0.05 L2seed2_s0.05 L2seed1_s0.05 L3_s0.05; do
  if [ "$r" = "L3_s0.05" ]; then until grep -q "^done" runs/L3_s0.05/qw.log; do sleep 30; done; fi
  ./runwhen.sh /home/agent/.venvs/aggext/bin/python qstruct.py runs/$r > runs/$r/qstruct.log 2>&1 &
  sleep 20
done
wait
