#!/bin/bash
# round 2: all 11 nets, 4 processes, OMP_NUM_THREADS=1
cd /home/agent/workspaces/mats/work/cm-interp
export OMP_NUM_THREADS=1
printf '%s\n' 3L_s10 3L_s11 2L_s0 2L_s1 3L_s12 3L_s13 2L_s2 2L_s3 3L_s14 2L_s4 plat_s3 | \
  xargs -P 4 -I{} sh -c '/home/agent/.venvs/aggext/bin/python r2run.py {} > r2/logs/{}.log 2>&1; echo "{} exit $?" >> r2/logs/run_all.status'
