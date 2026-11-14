#!/usr/bin/env bash
# E4 seed 2 (width chosen by the E3a rule) and the backward certificate (amendment 8), <= 4 active via runwhen.sh
cd /home/agent/workspaces/mats/work/cm-interp-B
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/agent/workspaces/mats/work/tiny-pfn
PY=/home/agent/.venvs/aggext/bin/python
( until grep -q "^done" runs/L2seed2_s0.05/qbayes.log 2>/dev/null; do sleep 30; done
  H=$($PY -c "import json; j=json.load(open('runs/L2seed2_s0.05/qbayes.json')); print(max(j['rows'], key=lambda r: r['R2']['mean'])['hidden'])")
  echo "E4 seed 2 width $H" > runs/L2seed2_s0.05/qpay_width.txt
  ./runwhen.sh $PY qpay.py runs/L2seed2_s0.05 e3a $H > runs/L2seed2_s0.05/qpay.log 2>&1 ) &
( until grep -q "^done" runs/L2_s0.05/qpay.log 2>/dev/null; do sleep 30; done
  ./runwhen.sh $PY bc.py pfn runs/L2_s0.05 --smoke > runs/L2_s0.05/bc_smoke.log 2>&1
  ./runwhen.sh $PY bc.py small 0.5 > results/bc_small_s0.5.log 2>&1 &
  sleep 30
  ./runwhen.sh $PY bc.py small 0.3 > results/bc_small_s0.3.log 2>&1 &
  sleep 30
  ./runwhen.sh $PY bc.py pfn runs/L2_s0.05 > runs/L2_s0.05/bc.log 2>&1 &
  wait ) &
wait
