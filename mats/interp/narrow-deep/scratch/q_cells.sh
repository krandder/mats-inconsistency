#!/bin/bash
cd /home/agent/workspaces/mats/work/narrow-deep
PY=/home/agent/.venvs/arc-noisy/bin/python
N9="nice -n 19 ionice -c3"
$N9 $PY scratch/p3_cells.py plain 16 64 0 0,1 runs/cells_plain_n16_L64_s0.json >> logs/cells.log 2>&1
$N9 $PY scratch/p3_cells.py res 8 64 0 0,1 runs/cells_res_n8_L64_s0.json >> logs/cells.log 2>&1
$N9 $PY scratch/p3_cells.py res 16 64 0 0,1 runs/cells_res_n16_L64_s0.json >> logs/cells.log 2>&1
$N9 $PY scratch/p3_cells.py plain 8 64 0.001 0,2 runs/cells_plain_n8_L64_s0.001.json >> logs/cells.log 2>&1
echo DONE > logs/q_cells.done
