#!/bin/bash
cd /home/agent/workspaces/mats/work/deep-study/wf7/sudoku
until grep -q "^exit" out/e_trustB_mB6.out 2>/dev/null; do sleep 3; done
until [ -f out/trust_meta_mA6.json ]; do sleep 2; done
./run_step.sh out/base_HARD.out e_search6.py base HARD || exit 1
./run_step.sh out/search_S4_HARD.out e_search6.py arm S4 HARD || exit 1
./run_step.sh out/search_S4_ALL.out e_search6.py arm S4 ALL || exit 1
./run_step.sh out/e_multi.out e_search6.py multi || exit 1
./run_step.sh out/search_S2_ALL.out e_search6.py arm S2 ALL 150 || exit 1
