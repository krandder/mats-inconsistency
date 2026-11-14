#!/bin/bash
cd /home/agent/workspaces/mats/work/deep-study/wf7/sudoku
until grep -q "^exit" out/e_4x4_mA4.out; do sleep 2; done
mv out/e_4x4_mA4.out out/e_4x4_mA4_dpfailed.out
./run_step.sh out/e_4x4_mA4.out e_4x4.py mA4
until [ -f out/trust_meta_mA6.json ]; do sleep 2; done
./run_step.sh out/e_trust6_analyse.out e_trust6.py mA6 analyse
./run_step.sh out/search_S1_HARD.out e_search6.py arm S1 HARD
./run_step.sh out/search_S1_ALL.out e_search6.py arm S1 ALL
./run_step.sh out/search_S5_HARD.out e_search6.py arm S5 HARD
./run_step.sh out/search_S2_HARD.out e_search6.py arm S2 HARD 100
