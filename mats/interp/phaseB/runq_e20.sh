#!/usr/bin/env bash
# E20 runs, sequential, lowest priority (Kelvin's rule); skips finished runs and waits for one already running
cd /home/agent/workspaces/mats/work/cm-interp-B
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/agent/workspaces/mats/work/tiny-pfn
while pgrep -f "python -W ignore minimal.py" >/dev/null; do sleep 15; done
for seed in 0 1 2; do
  for cfg in "ln 0.0" "pre 0.0" "ss 0.0" "ln 0.1" "pre 0.1"; do
    set -- $cfg
    [ -f results/e20/${1}_s${seed}_sig${2}.json ] && continue
    nice -n 19 ionice -c3 /home/agent/.venvs/aggext/bin/python -W ignore minimal.py $1 $seed --sigma $2
  done
done
