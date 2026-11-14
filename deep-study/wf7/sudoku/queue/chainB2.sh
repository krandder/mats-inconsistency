#!/bin/bash
cd /home/agent/workspaces/mats/work/deep-study/wf7/sudoku
until grep -q "^exit" out/e_4x4_mA4_nodp.out; do sleep 2; done
./run_step.sh out/train_mB6t.out train_m.py mB6t B 6 32 2 3000 256 12 data/k6_train_min.npz || exit 1
./run_step.sh out/e_4x4_mA4.out e_4x4.py mA4 || exit 1
until [ -f out/trust_meta_mA6.json ]; do sleep 2; done
./run_step.sh out/e_trust6_analyse.out e_trust6.py mA6 analyse || exit 1
./run_step.sh out/search_S1_HARD.out e_search6.py arm S1 HARD || exit 1
./run_step.sh out/search_S1_ALL.out e_search6.py arm S1 ALL || exit 1
./run_step.sh out/search_S5_HARD.out e_search6.py arm S5 HARD || exit 1
./run_step.sh out/search_S2_HARD.out e_search6.py arm S2 HARD 100 || exit 1
