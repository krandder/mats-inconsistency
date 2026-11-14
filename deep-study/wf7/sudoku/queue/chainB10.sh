#!/bin/bash
cd /home/agent/workspaces/mats/work/deep-study/wf7/sudoku
until grep -q "^exit" out/search_S1_ALL_T2.out 2>/dev/null; do sleep 3; done
./run_step.sh out/e_multi.out e_search6.py multi 40 64
