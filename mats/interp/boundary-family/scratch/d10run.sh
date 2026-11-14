#!/bin/bash
# Kelvin item 1: oracle cells across dev nets; args: list of seeds (run sequentially)
cd /home/agent/workspaces/mats/work/boundary-family
for s in "$@"; do
  TH=3 nice -n 19 ionice -c3 /home/agent/.venvs/arc-noisy/bin/python scratch/diag10.py $s 19 20 > logs/diag10_${s}.log 2>&1
done
