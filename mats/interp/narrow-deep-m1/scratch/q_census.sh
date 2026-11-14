#!/bin/bash
cd /home/agent/workspaces/mats/work/narrow-deep-m1
PY=/home/agent/.venvs/arc-noisy/bin/python
nice -n 19 ionice -c3 $PY scratch/census.py $1 $2 runs/census_$1_$2.json > logs/census_$1_$2.log 2>&1
echo DONE > logs/q_census_$1_$2.done
