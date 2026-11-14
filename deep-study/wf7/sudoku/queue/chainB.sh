#!/bin/bash
cd /home/agent/workspaces/mats/work/deep-study/wf7/sudoku
until grep -q "^exit" out/train_mA4.out; do sleep 2; done
./run_step.sh out/e_4x4_mA4_nodp.out e_4x4.py mA4 skipdp || exit 1
./run_step.sh out/train_mB6t.out train_m.py mB6t B 6 32 2 3000 256 12 data/k6_train_min.npz || exit 1
./run_step.sh out/e_4x4_mA4.out e_4x4.py mA4 || exit 1
