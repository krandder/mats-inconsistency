#!/bin/bash
# Post-process truth runs as they appear: deep-part oracle (p3_att) then samplers (p3_samplers). One process.
cd /home/agent/workspaces/mats/work/narrow-deep
PY=/home/agent/.venvs/arc-noisy/bin/python
N9="nice -n 19 ionice -c3"
while [ ! -f logs/q_cells.done ]; do sleep 20; done
for pass in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40; do
  did=0
  for j in runs/p3/truth_*.json; do
    case $j in *_att.json|*_samplers.json) continue;; esac
    p=${j%.json}
    [ -f ${p}_oracle.npz ] || continue
    if [ ! -f ${p}_att.json ]; then $N9 $PY scratch/p3_att.py $p >> logs/p3_att.log 2>&1; did=1; fi
    if [ ! -f ${p}_samplers.json ]; then $N9 $PY scratch/p3_samplers.py $p 32 >> logs/p3_samplers.log 2>&1; did=1; fi
  done
  if [ -f logs/q_truth_a.done ] && [ -f logs/q_truth_b.done ] && [ $did = 0 ]; then break; fi
  [ $did = 0 ] && sleep 60
done
echo DONE > logs/q_p3post.done
