#!/bin/bash
# factorized K=3 (and K=2) vs MC at depth n/8 and n/4; one process, nice/ionice; waits for the K=2 sweep to finish
PY=/home/agent/.venvs/arc-noisy/bin/python
cd "$(dirname "$0")"
# (runs concurrently with the K=2 sweep; 2 processes total)
for spec in "32 17" "64 17" "128 17"; do
  set -- $spec
  for seed in $(seq 900 907); do
    nice -n 19 ionice -c3 $PY k3f_scaling.py $1 $seed $2 4 >> logs/k3f.log 2>&1
  done
done
for seed in 900 901 902; do
  nice -n 19 ionice -c3 $PY k3f_scaling.py 256 $seed 16 4 >> logs/k3f.log 2>&1
done
echo DONE3 >> logs/k3f.log
