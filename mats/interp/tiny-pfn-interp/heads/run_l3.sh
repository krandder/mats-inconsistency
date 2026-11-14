#!/usr/bin/env bash
# Amendment 6: run l3.py on L3 seeds 10-14 and base seeds 10,12,13,14, at most 4 at a time (CPU).
cd /home/agent/workspaces/mats/work/tiny-pfn-interp/heads
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/agent/.venvs/aggext/bin/python
mkdir -p l3out
one() { nice -n 5 $PY l3.py hruns/$1_ws_s$2 $1_s$2 l3out/$1_s$2.json > l3out/$1_s$2.log 2>&1; echo "done $1 $2 tracebacks=$(grep -c Traceback l3out/$1_s$2.log)"; }
one L3 10 & one L3 11 & one L3 12 & one L3 13 & wait
one L3 14 & one base 10 & one base 12 & one base 13 & wait
one base 14
echo L3-ALL-DONE
