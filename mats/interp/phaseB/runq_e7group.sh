#!/usr/bin/env bash
cd /home/agent/workspaces/mats/work/cm-interp-B
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/agent/workspaces/mats/work/tiny-pfn
PY=/home/agent/.venvs/aggext/bin/python
for r in L2mx2_s0.05 L2nb5_s0.05 L2nb6_s0.05 L2nb7_s0.05; do
  ( until [ -f runs/$r/meta.json ]; do sleep 30; done
    ./runwhen.sh $PY qgroup.py predict runs/$r > runs/$r/qgroup.log 2>&1
    ./runwhen.sh $PY qshare.py dev runs/$r > runs/$r/share.log 2>&1 ) &
  sleep 5
done
wait
