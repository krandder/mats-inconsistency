#!/usr/bin/env bash
cd /home/agent/workspaces/mats/work/cm-interp-B
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/agent/workspaces/mats/work/tiny-pfn
PY=/home/agent/.venvs/aggext/bin/python
for r in L2_s0.05 L2seed1_s0.05 L2seed2_s0.05; do ./runwhen.sh $PY qshare.py dev runs/$r > runs/$r/share.log 2>&1 & sleep 20; done
for spec in "L2ws0_s0.05 --fit" "L2ws3_s0.05 --fit" "L2mx0_s0.05" "L2mx1_s0.05" "L2mx4_s0.05"; do
  set -- $spec
  ( until [ -f runs/$1/meta.json ]; do sleep 30; done; ./runwhen.sh $PY qshare.py dev runs/$1 $2 > runs/$1/share.log 2>&1 ) &
  sleep 5
done
wait
