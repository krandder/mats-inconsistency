#!/usr/bin/env bash
# Regenerate every table in the report from raw results (truth/, results/, samp/, samp_v2/).
cd "$(dirname "$0")"
PY=/home/agent/.venvs/arc-noisy/bin/python
E=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23
export OMP_NUM_THREADS=2
nice -n 15 $PY analyze.py > results/tables.md 2>&1
{
  for s in 0.001 0.003; do for L in 8 32; do nice -n 15 $PY hybrid.py $E $s $L; done; done
} > results/h1.md 2>&1
{
  for s in 0.001 0.003 0.01 0.03 0.1; do for L in 8 32; do nice -n 15 $PY combine.py $s $L; echo; done; done
} > results/h2.md 2>&1
nice -n 15 $PY trend.py > results/trend.md 2>&1
echo done
