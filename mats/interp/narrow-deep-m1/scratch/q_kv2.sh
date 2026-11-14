#!/bin/bash
cd /home/agent/workspaces/mats/work/narrow-deep-m1
PY=/home/agent/.venvs/arc-noisy/bin/python
for S in "$@"; do
  nice -n 19 ionice -c3 $PY scratch/oracle_k_by_cut.py $S 0.001 runs/kbycut2_s${S}_sig0.001.json 15 >> logs/kbycut2_$1.log 2>&1
done
echo DONE > logs/q_kv2_$1.done
