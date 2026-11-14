#!/bin/bash
cd /home/agent/workspaces/mats/work/deep-study/wf7/sudoku
until grep -q "^exit" out/gen_puzzles_train.out; do sleep 2; done
./run_step.sh out/train_mB6.out train_m.py mB6 B 6 64 4 3000 256 12 data/k6_train_min.npz || exit 1
./run_step.sh out/e_trustB_mB6.out e_trustB.py mB6 || exit 1
