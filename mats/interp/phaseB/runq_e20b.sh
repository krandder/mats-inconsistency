#!/usr/bin/env bash
cd /home/agent/workspaces/mats/work/cm-interp-B
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/agent/workspaces/mats/work/tiny-pfn
for seed in 0 1 2; do for sg in 0.0 0.1; do
  [ -f results/e20/lnmean_s${seed}_sig${sg}.json ] && continue
  nice -n 19 ionice -c3 /home/agent/.venvs/aggext/bin/python -W ignore minimal.py lnmean $seed --sigma $sg
done; done
