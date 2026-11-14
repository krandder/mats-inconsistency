#!/usr/bin/env bash
# Master queue (E2 3L, E3a, E3b), <= 4 active analysis processes via runwhen.sh.
cd /home/agent/workspaces/mats/work/cm-interp-B
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/agent/workspaces/mats/work/tiny-pfn
PY=/home/agent/.venvs/aggext/bin/python
# chain 1: after the 3L E1 run, E3a on all four models, sequentially
( until grep -q "^done" runs/L3_s0.05/qw.log; do sleep 30; done
  for r in L2_s0.05 L2seed1_s0.05 L2seed2_s0.05 L3_s0.05; do ./runwhen.sh $PY qbayes.py runs/$r > runs/$r/qbayes.log 2>&1; done ) &
sleep 5
# chain 2: after the first 2L E2 run finishes, E2 on 3L then E3b on 2L seed 4 and 3L
( until grep -q "^done" runs/L2_s0.05/qstruct.log; do sleep 30; done
  until grep -q "^done" runs/L3_s0.05/qw.log; do sleep 30; done
  ./runwhen.sh $PY qstruct.py runs/L3_s0.05 > runs/L3_s0.05/qstruct.log 2>&1
  ./runwhen.sh $PY qstruct_b.py runs/L3_s0.05 > runs/L3_s0.05/qstruct_b.log 2>&1 ) &
sleep 5
# chain 3: after the seed-2 E2 run, E3b on seed 4 then seed 2
( until grep -q "^done" runs/L2seed2_s0.05/qstruct.log; do sleep 30; done
  ./runwhen.sh $PY qstruct_b.py runs/L2_s0.05 > runs/L2_s0.05/qstruct_b.log 2>&1
  ./runwhen.sh $PY qstruct_b.py runs/L2seed2_s0.05 > runs/L2seed2_s0.05/qstruct_b.log 2>&1 ) &
sleep 5
# chain 4: after the seed-1 E2 run, E3b on seed 1
( until grep -q "^done" runs/L2seed1_s0.05/qstruct.log; do sleep 30; done
  ./runwhen.sh $PY qstruct_b.py runs/L2seed1_s0.05 > runs/L2seed1_s0.05/qstruct_b.log 2>&1 ) &
wait
