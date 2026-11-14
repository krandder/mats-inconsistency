#!/bin/bash
cd /home/agent/workspaces/mats/work/deep-study/wf7/sudoku
until grep -q "^exit" out/train_D6.out 2>/dev/null; do sleep 3; done
until grep -q "^exit" out/e_trustB_mB6.out 2>/dev/null; do sleep 3; done
./run_step.sh out/base_ALL.out e_search6.py base ALL || exit 1
./run_step.sh out/base_T12.out e_search6.py base T12 || exit 1
./run_step.sh out/search_S3_HARD.out e_search6.py arm S3 HARD || exit 1
./run_step.sh out/search_S3_ALL.out e_search6.py arm S3 ALL || exit 1
./run_step.sh out/search_S3_T12.out e_search6.py arm S3 T12 || exit 1
./run_step.sh out/search_S1_T12.out e_search6.py arm S1 T12 || exit 1
