#!/bin/bash
# sequential sweep, one process, nice/ionice; K=2 vs MC along L proportional to n
PY=/home/agent/.venvs/arc-noisy/bin/python
cd "$(dirname "$0")"
for spec in "32 16 17" "64 32 17" "128 64 17" "256 128 16" "512 256 16"; do
  set -- $spec
  for seed in 900 901 902 903; do
    nice -n 19 ionice -c3 $PY k2_scaling.py $1 $2 $seed $3 4 >> logs/k2s.log 2>&1
  done
done
nice -n 19 ionice -c3 $PY k2_scaling.py 1024 256 900 15 4 >> logs/k2s.log 2>&1
echo DONE >> logs/k2s.log
