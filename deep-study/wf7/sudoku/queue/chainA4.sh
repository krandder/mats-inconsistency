#!/bin/bash
cd /home/agent/workspaces/mats/work/deep-study/wf7/sudoku
until grep -q "^exit" out/e_trustB_mB6.out 2>/dev/null; do sleep 3; done
./run_step.sh out/e_stiff.out e_stiff.py
./run_step.sh out/base_HARD.out e_search6.py base HARD
./run_step.sh out/search_S4_HARD.out e_search6.py arm S4 HARD
./run_step.sh out/search_S4_HARD_T2.out e_search6.py arm S4 HARD 300 T2
./run_step.sh out/search_S4_ALL.out e_search6.py arm S4 ALL
./run_step.sh out/e_multi.out e_search6.py multi
./run_step.sh out/e_4x4_mA4_order.out e_4x4.py mA4 order
