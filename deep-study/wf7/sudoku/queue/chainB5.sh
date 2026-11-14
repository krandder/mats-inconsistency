#!/bin/bash
cd /home/agent/workspaces/mats/work/deep-study/wf7/sudoku
./run_step.sh out/e_posthoc.out e_posthoc.py
./run_step.sh out/search_S1_HARD_T2.out e_search6.py arm S1 HARD 300 T2
./run_step.sh out/e_trust6_analyse.out e_trust6.py mA6 analyse
./run_step.sh out/search_S2_HARD.out e_search6.py arm S2 HARD 100
./run_step.sh out/search_S1_ALL.out e_search6.py arm S1 ALL
./run_step.sh out/search_S2_HARD_T2.out e_search6.py arm S2 HARD 100 T2
