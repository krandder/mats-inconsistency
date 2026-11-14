#!/bin/bash
cd /home/agent/workspaces/mats/work/deep-study/wf7/sudoku
./run_step.sh out/search_S4_HARD.out e_search6.py arm S4 HARD 100
./run_step.sh out/e_multi.out e_search6.py multi
./run_step.sh out/search_S4_HARD_T2.out e_search6.py arm S4 HARD 60 T2
./run_step.sh out/e_4x4_mA4_order.out e_4x4.py mA4 order
./run_step.sh out/search_S4_ALL.out e_search6.py arm S4 ALL 200
