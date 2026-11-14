#!/bin/bash
# second sweep: more seeds, larger widths; one process, nice/ionice
PY=/home/agent/.venvs/arc-noisy/bin/python
cd "$(dirname "$0")"
for spec in "32 16 17" "64 32 17" "128 64 17" "256 128 16" "512 256 16"; do
  set -- $spec
  for seed in $(seq 904 915); do
    nice -n 19 ionice -c3 $PY k2_scaling.py $1 $2 $seed $3 4 >> logs/k2s.log 2>&1
  done
done
for seed in $(seq 900 907); do
  nice -n 19 ionice -c3 $PY k2_scaling.py 1024 512 $seed 15 4 >> logs/k2s.log 2>&1
done
for seed in $(seq 900 903); do
  nice -n 19 ionice -c3 $PY k2_scaling.py 2048 512 $seed 14 4 >> logs/k2s.log 2>&1
done
echo DONE2 >> logs/k2s.log
