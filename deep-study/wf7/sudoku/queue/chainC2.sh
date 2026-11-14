#!/bin/bash
cd /home/agent/workspaces/mats/work/deep-study/wf7/sudoku
until grep -q "^exit" out/train_mA6.out; do sleep 2; done
./run_step.sh out/e_trust6_fit.out e_trust6.py mA6 fit || exit 1
./run_step.sh out/e_distill.out e_distill.py || exit 1
./run_step.sh out/train_D6.out train_m.py D6 B 6 64 4 2000 256 13 data/k6_train_12p.npz out/distill_labels.npz || exit 1
