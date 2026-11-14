#!/bin/bash
cd /home/agent/workspaces/mats/work/deep-study/wf7/sudoku
until grep -q "^exit" out/e_trust6_analyse.out 2>/dev/null; do sleep 3; done
./run_step.sh out/search_S2_HARD.out e_search6.py arm S2 HARD 60
./run_step.sh out/search_S2_HARD_T2.out e_search6.py arm S2 HARD 60 T2
./run_step.sh out/search_S1_ALL.out e_search6.py arm S1 ALL
./run_step.sh out/search_S1_ALL_T2.out e_search6.py arm S1 ALL 500 T2
