#!/bin/bash
# registered runs, first batch (three processes): Sudoku seed 1, parity worlds 102 and 103
cd /home/agent/workspaces/mats/work/deep-study/wf13/loop
PY="nice -n 19 ionice -c3 /home/agent/.venvs/aggext/bin/python -B"
$PY srun.py 1 > logs/chain_s1.out 2>&1 &
$PY prun.py 102 1 > logs/chain_p102.out 2>&1 &
$PY prun.py 103 1 > logs/chain_p103.out 2>&1 &
wait
echo batch1 done $(date -u +%H:%M:%S) >> logs/CHAIN_LOG
