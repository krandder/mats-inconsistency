#!/usr/bin/env bash
# E50 after the VM (fifth agent): collect runs into runs/e50/, evaluate them with e50_eval.py through the CPU queue (4 processes,
# nice 19), then the summary.
cd /home/agent/workspaces/mats/work/cm-interp-B
./gpu/collect_b5.sh cmb-e50-1
PY=/home/agent/.venvs/aggext/bin/python
$PY mkq.py e50_eval.py runs/e50 results/e50 4 /tmp/q_e50.txt
./cpuq.sh /tmp/q_e50.txt results/e50_eval.log 4
nice -n 19 ionice -c3 $PY e50_summ.py > results/e50_summ.out 2>&1
touch /tmp/e50_done
