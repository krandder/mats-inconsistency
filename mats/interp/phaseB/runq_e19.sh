#!/usr/bin/env bash
# E19: start the post-MLP trainings as slots free (at most 3 train_var.py processes at once); lowest priority (Kelvin's rule)
cd /home/agent/workspaces/mats/work/cm-interp-B
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/agent/workspaces/mats/work/tiny-pfn
for s in 5 6 7; do
  [ -d runs/var_post_s$s ] && continue
  while [ "$(pgrep -f 'python -W ignore train_var.py' | wc -l)" -ge 3 ]; do sleep 30; done
  nohup nice -n 19 ionice -c3 /home/agent/.venvs/aggext/bin/python -W ignore train_var.py post --seed $s --out runs/var_post_s$s > runs/var_post_s$s.log 2>&1 &
  sleep 60
done
wait
