#!/usr/bin/env bash
# E41 (PREREG-B amendment 49): the toy's format and switch arms, seeds 0-11, at most 4 processes, lowest CPU and I/O priority.
cd /home/agent/workspaces/mats/work/cm-interp-B
export OMP_NUM_THREADS=1
for S in 0 1 2 3 4 5 6 7 8 9 10 11; do
  for ARM in ctrl mag c2m2k c2m12k mag18 m2c12k; do
    while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 5; done
    [ -f results/e41/${ARM}_s${S}.json ] || nice -n 19 ionice -c3 /home/agent/.venvs/aggext/bin/python toy_fmt.py $ARM $S >> results/e41_run.log 2>&1 &
  done
done
wait
echo E41-DONE >> results/e41_run.log
