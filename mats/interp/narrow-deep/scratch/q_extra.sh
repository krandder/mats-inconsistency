#!/bin/bash
cd /home/agent/workspaces/mats/work/narrow-deep
PY=/home/agent/.venvs/arc-noisy/bin/python
N9="nice -n 19 ionice -c3"
$N9 $PY scratch/p3_k2variants.py runs/p3/truth_res_n16_L64_s0.001_seed0 runs/p3/truth_res_n16_L64_s0.001_seed1 runs/p3/truth_res_n16_L256_s0.001_seed0 runs/p3/truth_res_n8_L64_s0.001_seed0 runs/p3/truth_plain_n16_L256_s0.001_seed0 runs/p3/truth_plain_n16_L64_s0.001_seed1 > logs/k2variants.log 2>&1
$N9 $PY scratch/p3_gauss.py res 16 256 0.001 0,1 runs/gauss_res_n16_L256.json > logs/gauss.log 2>&1
$N9 $PY scratch/p3_gauss.py res 8 256 0.001 0,1 runs/gauss_res_n8_L256.json >> logs/gauss.log 2>&1
$N9 $PY scratch/p3_gauss.py plain 16 256 0.001 0,1 runs/gauss_plain_n16_L256.json >> logs/gauss.log 2>&1
$N9 $PY scratch/check_rb.py > logs/check_rb.log 2>&1
echo DONE > logs/q_extra.done
