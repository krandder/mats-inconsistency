#!/bin/bash
# E27 runs (PREREG-B amendment 35): lowest-priority CPU chains. usage: bash runq_e27.sh CHAIN (a|b|c)
cd /home/agent/workspaces/mats/work/cm-interp-B
export OMP_NUM_THREADS=1
PY=/home/agent/.venvs/aggext/bin/python
run() { nice -n 19 ionice -c3 $PY e27_run.py "$@" >> results/e27_run_$CHAIN.log 2>&1; }
CHAIN=$1
case $CHAIN in
  a) [ -f results/e27_data.npz ] || nice -n 19 ionice -c3 $PY e27_data.py >> results/e27_run_a.log 2>&1; run M 0 15; run G 0 11 ;;
  b) run G 11 22 ;;
  c) run G 22 33 ;;
esac
echo "chain $CHAIN done $(date -u)" >> results/e27_run_$CHAIN.log
