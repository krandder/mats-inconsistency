#!/bin/bash
cd /home/agent/workspaces/mats/work/narrow-deep
PY=/home/agent/.venvs/arc-noisy/bin/python
for sig in 0.001 0; do
  nice -n 19 ionice -c3 $PY scratch/p2_jac.py plain 16 256 $sig 0,1,2,3 runs/jac_plain_n16_L256_s${sig}.json > logs/jac_plain_n16_L256_s${sig}.log 2>&1
  nice -n 19 ionice -c3 $PY scratch/p2_jac.py plain 16 64 $sig 0,1,2,3 runs/jac_plain_n16_L64_s${sig}.json > logs/jac_plain_n16_L64_s${sig}.log 2>&1
done
echo DONE > logs/q_jac_v2a.done
