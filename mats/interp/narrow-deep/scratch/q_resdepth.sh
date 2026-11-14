#!/bin/bash
cd /home/agent/workspaces/mats/work/narrow-deep
PY=/home/agent/.venvs/arc-noisy/bin/python
N9="nice -n 19 ionice -c3"
while [ ! -f logs/q_truth_b.done ]; do sleep 30; done
$N9 $PY scratch/p3_resdepth.py res 16 16 0.001 0,1,2,3 21 runs/resdepth_n16_L16.json >> logs/resdepth.log 2>&1
$N9 $PY scratch/p3_resdepth.py res 8 16 0.001 0,1,2,3 21 runs/resdepth_n8_L16.json >> logs/resdepth.log 2>&1
$N9 $PY scratch/p3_resdepth.py res 16 64 0.001 0,1,2,3 21 runs/resdepth_n16_L64.json >> logs/resdepth.log 2>&1
$N9 $PY scratch/p3_resdepth.py res 8 64 0.001 0,1,2,3 21 runs/resdepth_n8_L64.json >> logs/resdepth.log 2>&1
$N9 $PY scratch/p3_resdepth.py res 16 256 0.001 0,1 21 runs/resdepth_n16_L256.json >> logs/resdepth.log 2>&1
$N9 $PY scratch/p3_resdepth.py res 8 256 0.001 0,1 21 runs/resdepth_n8_L256.json >> logs/resdepth.log 2>&1
$N9 $PY scratch/p3_resdepth.py res 16 1024 0.001 0,1 20 runs/resdepth_n16_L1024.json >> logs/resdepth.log 2>&1
$N9 $PY scratch/p3_resdepth.py res 8 1024 0.001 0,1 20 runs/resdepth_n8_L1024.json >> logs/resdepth.log 2>&1
echo DONE > logs/q_resdepth.done
