#!/bin/bash
cd /home/agent/workspaces/mats/work/deep-study/wf7/sudoku
./run_step.sh out/search_S4_HARD_T2.out e_search6.py arm S4 HARD 60 T2
./run_step.sh out/search_S3_HARD_T2.out e_search6.py arm S3 HARD 60 T2
./run_step.sh out/search_S1_ALL_T2.out e_search6.py arm S1 ALL 500 T2
