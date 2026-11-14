#!/bin/bash
# After truth queue A: linearity shares (all random-net configs), then tails. One process.
cd /home/agent/workspaces/mats/work/narrow-deep
PY=/home/agent/.venvs/arc-noisy/bin/python
N9="nice -n 19 ionice -c3"
while [ ! -f logs/q_truth_a.done ]; do sleep 30; done
$N9 $PY scratch/p3_linshare.py res 16 64 0.001 0,1,2,3 runs/linshare_res_n16_L64.json >> logs/linshare.log 2>&1
$N9 $PY scratch/p3_linshare.py res 16 256 0.001 0,1,2,3 runs/linshare_res_n16_L256.json >> logs/linshare.log 2>&1
$N9 $PY scratch/p3_linshare.py res 8 64 0.001 0,1,2,3 runs/linshare_res_n8_L64.json >> logs/linshare.log 2>&1
$N9 $PY scratch/p3_linshare.py res 8 256 0.001 0,1,2,3 runs/linshare_res_n8_L256.json >> logs/linshare.log 2>&1
$N9 $PY scratch/p3_linshare.py plain 16 64 0.001 0,1,2,3 runs/linshare_plain_n16_L64.json >> logs/linshare.log 2>&1
$N9 $PY scratch/p3_linshare.py plain 16 256 0.001 0,1,2,3 runs/linshare_plain_n16_L256.json >> logs/linshare.log 2>&1
$N9 $PY scratch/p3_linshare.py plain 8 64 0.001 0,2,3,4 runs/linshare_plain_n8_L64.json >> logs/linshare.log 2>&1
for p in runs/p3/truth_res_n16_L64_s0.001_seed0 runs/p3/truth_res_n16_L64_s0.001_seed1 runs/p3/truth_plain_n16_L256_s0.001_seed0 runs/p3/truth_plain_n16_L256_s0.001_seed1 runs/p3/truth_plain_n16_L64_s0.001_seed0 runs/p3/truth_plain_n8_L64_s0.001_seed0 runs/p3/truth_plain_n8_L256_s0.001_seed2; do
  [ -f $p.json ] && [ ! -f ${p}_tails.json ] && $N9 $PY scratch/p3_tails.py $p 16 >> logs/p3_tails.log 2>&1
done
while [ ! -f logs/q_truth_b.done ]; do sleep 30; done
for p in runs/p3/truth_res_n16_L256_s0.001_seed0 runs/p3/truth_res_n8_L64_s0.001_seed0 runs/p3/truth_res_n8_L256_s0.001_seed0; do
  [ -f $p.json ] && [ ! -f ${p}_tails.json ] && $N9 $PY scratch/p3_tails.py $p 16 >> logs/p3_tails.log 2>&1
done
echo DONE > logs/q_tails.done
