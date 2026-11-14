#!/bin/bash
cd /home/agent/workspaces/mats/work/deep-study/wf7/sudoku
until grep -q "^exit" out/search_S3_HARD.out 2>/dev/null; do sleep 3; done
./run_step.sh out/base_ALL.out e_search6.py base ALL
./run_step.sh out/base_T12.out e_search6.py base T12
./run_step.sh out/search_S3_T12.out e_search6.py arm S3 T12
./run_step.sh out/search_S3_ALL.out e_search6.py arm S3 ALL 200
./run_step.sh out/search_S3_HARD_T2.out e_search6.py arm S3 HARD 100 T2
./run_step.sh out/search_S1_T12.out e_search6.py arm S1 T12
