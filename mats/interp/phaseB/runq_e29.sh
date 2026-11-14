#!/bin/bash
# E29 runs (PREREG-B amendment 37): lowest-priority CPU chains. usage: bash runq_e29.sh CHAIN (a|b|c)
cd /home/agent/workspaces/mats/work/cm-interp-B
export OMP_NUM_THREADS=1
PY=/home/agent/.venvs/aggext/bin/python
run() { nice -n 19 ionice -c3 $PY e29_run.py "$@" >> results/e29_run_$CHAIN.log 2>&1; }
CHAIN=$1
until [ -f results/e29_data.npz ]; do sleep 5; done
case $CHAIN in
  a) run M 0 15 ;;
  b) run G 0 16 ;;
  c) run G 16 33 ;;
esac
echo "chain $CHAIN done $(date -u)" >> results/e29_run_$CHAIN.log
