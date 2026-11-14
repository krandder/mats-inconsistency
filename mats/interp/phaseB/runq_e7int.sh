#!/usr/bin/env bash
cd /home/agent/workspaces/mats/work/cm-interp-B
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/agent/workspaces/mats/work/tiny-pfn
PY=/home/agent/.venvs/aggext/bin/python
( for r in L2_s0.05 L2seed2_s0.05 L2ws0_s0.05 L2mx4_s0.05; do ./runwhen.sh $PY qint.py runs/$r > runs/$r/qint.log 2>&1; done ) &
sleep 30
( for r in L2seed1_s0.05 L2mx0_s0.05 L2ws3_s0.05 L2mx1_s0.05; do ./runwhen.sh $PY qint.py runs/$r > runs/$r/qint.log 2>&1; done ) &
wait
kill -CONT 2051055 2046300 2>/dev/null
echo "resumed seed-7 base at $(date -u)"
