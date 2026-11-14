#!/bin/bash
cd /home/agent/workspaces/mats/work/narrow-deep
PY=/home/agent/.venvs/arc-noisy/bin/python
while [ ! -f logs/q_truth_a.done ]; do sleep 30; done
for p in runs/p3/truth_plain_n16_L256_s0.001_seed0 runs/p3/truth_plain_n16_L256_s0.001_seed1 runs/p3/truth_plain_n8_L256_s0.001_seed2 runs/p3/truth_plain_n8_L256_s0.001_seed4 runs/p3/truth_plain_n16_L64_s0.001_seed0 runs/p3/truth_plain_n16_L64_s0.001_seed1 runs/p3/truth_plain_n8_L64_s0.001_seed0 runs/p3/truth_plain_n8_L64_s0.001_seed2; do
  [ -f $p.json ] && [ ! -f ${p}_tails.json ] && nice -n 19 ionice -c3 $PY scratch/p3_tails.py $p 16 >> logs/p3_tails.log 2>&1
done
while [ ! -f logs/q_truth_b.done ]; do sleep 30; done
for p in runs/p3/truth_res_n16_L64_s0.001_seed0 runs/p3/truth_res_n16_L64_s0.001_seed1 runs/p3/truth_res_n16_L256_s0.001_seed0 runs/p3/truth_res_n8_L64_s0.001_seed0 runs/p3/truth_res_n8_L256_s0.001_seed0; do
  [ -f $p.json ] && [ ! -f ${p}_tails.json ] && nice -n 19 ionice -c3 $PY scratch/p3_tails.py $p 16 >> logs/p3_tails.log 2>&1
done
echo DONE > logs/q_tails.done
