#!/usr/bin/env bash
# Generate every val/test set (exact Bayes) for the pre-registered worlds. Two parallel lanes, lowest priority.
cd /home/agent/workspaces/mats/work/scale-free-exp
export OMP_NUM_THREADS=1
P="nice -n 19 ionice -c3 /home/agent/.venvs/aggext/bin/python sftrain.py mkdata"
(
  for N in 3 6 12 24; do for sp in val test; do $P --world indep --N $N --split $sp --n 4096; done; done
  for N in 4 6 8; do for sp in val test; do $P --world pairs --N $N --rho 0.9 --split $sp --n 4096; done; done
) > data_lane1.log 2>&1 &
(
  for N in 4 8 16 24; do for sp in val test; do $P --world groups --N $N --rho 0.5 --gsize 4 --split $sp --n 4096; done; done
) > data_lane2.log 2>&1 &
wait
echo DATA-DONE
