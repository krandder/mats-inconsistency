#!/bin/bash
cd /home/agent/workspaces/mats/work/deep-study/wf7/sudoku
until grep -q "^exit" out/search_S2_HARD.out 2>/dev/null; do sleep 3; done
./run_step.sh out/search_S1_ALL.out e_search6.py arm S1 ALL
./run_step.sh out/search_S1_ALL_T2.out e_search6.py arm S1 ALL 500 T2
