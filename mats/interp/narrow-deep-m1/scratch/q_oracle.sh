#!/bin/bash
cd /home/agent/workspaces/mats/work/narrow-deep-m1
PY=/home/agent/.venvs/arc-noisy/bin/python
for spec in "$@"; do set -- $spec
  nice -n 19 ionice -c3 $PY scratch/oracle_structure.py $1 $2 runs/oracle_structure_s$1_sig$2.json 16 > logs/oracle_s$1_sig$2.log 2>&1
done
echo DONE > logs/q_oracle_$1.done
