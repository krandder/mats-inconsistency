#!/usr/bin/env bash
cd /home/agent/workspaces/mats/work/cm-interp-B
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/agent/workspaces/mats/work/tiny-pfn
PY=/home/agent/.venvs/aggext/bin/python
for r in L2_s0.05 L2seed1_s0.05 L2seed2_s0.05; do ./runwhen.sh $PY qsys.py runs/$r > runs/$r/qsys.log 2>&1 & sleep 25; done
( until grep -q "^done" runs/L3_s0.05/qstruct_b.log 2>/dev/null; do sleep 30; done; ./runwhen.sh $PY qsys.py runs/L3_s0.05 > runs/L3_s0.05/qsys.log 2>&1 ) &
wait
