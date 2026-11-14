#!/bin/bash
# E26 runs (PREREG-B amendment 34): four lowest-priority CPU chains. usage: bash runq_e26.sh CHAIN (a|b|c|d)
cd /home/agent/workspaces/mats/work/cm-interp-B
export OMP_NUM_THREADS=1
PY=/home/agent/.venvs/aggext/bin/python
run() { nice -n 19 ionice -c3 $PY e26_run.py "$@" >> results/e26_run_$CHAIN.log 2>&1; }
CHAIN=$1
case $CHAIN in
  a) run M 0 15; run G 0 33; run NZ 0 4 ;;
  b) run L1 0 20; run HD 0 20; run NZ 4 8 ;;
  c) run DY5 0 60; run NZ 8 10 ;;
  d) run DY6 0 59; run NZ 10 12 ;;
esac
echo "chain $CHAIN done $(date -u)" >> results/e26_run_$CHAIN.log
