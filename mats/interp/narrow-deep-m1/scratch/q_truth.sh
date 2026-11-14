#!/bin/bash
# usage: q_truth.sh NAME [wait-for-file] -- "SEED SIG LOG2P" ...
cd /home/agent/workspaces/mats/work/narrow-deep-m1
PY=/home/agent/.venvs/arc-noisy/bin/python
NAME=$1; shift
if [ "$1" != "--" ]; then while [ ! -f "$1" ]; do sleep 20; done; shift; fi
shift
for spec in "$@"; do set -- $spec
  nice -n 19 ionice -c3 $PY scratch/truth.py $1 $2 runs/truth_s$1_sig$2 $3 >> logs/truth_$NAME.log 2>&1
done
echo DONE > logs/q_truth_$NAME.done
