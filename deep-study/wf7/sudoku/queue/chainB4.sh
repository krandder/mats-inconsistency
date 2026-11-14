#!/bin/bash
cd /home/agent/workspaces/mats/work/deep-study/wf7/sudoku
./run_step.sh out/search_S1_HARD.out e_search6.py arm S1 HARD
./run_step.sh out/e_trust6_analyse.out e_trust6.py mA6 analyse
./run_step.sh out/search_S1_ALL.out e_search6.py arm S1 ALL
./run_step.sh out/search_S5_HARD.out e_search6.py arm S5 HARD
./run_step.sh out/search_S2_HARD.out e_search6.py arm S2 HARD 100
