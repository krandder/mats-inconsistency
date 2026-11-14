#!/bin/bash
cd /home/agent/workspaces/mats/work/deep-study/wf7/sudoku
until grep -q "^exit" out/search_S4_HARD.out 2>/dev/null; do sleep 3; done
./run_step.sh out/e_multi.out e_search6.py multi
./run_step.sh out/e_4x4_mA4_order.out e_4x4.py mA4 order
