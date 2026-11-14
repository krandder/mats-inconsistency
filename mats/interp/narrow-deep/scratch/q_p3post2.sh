#!/bin/bash
# Post-process truth runs (trimmed, Amendment 2): p3_att on every truth; p3_samplers (R=16) on two seeds per config,
# residual configs first (coordinator's headline). One process.
cd /home/agent/workspaces/mats/work/narrow-deep
PY=/home/agent/.venvs/arc-noisy/bin/python
N9="nice -n 19 ionice -c3"
SAMP="res_n16_L64_s0.001_seed0 res_n16_L64_s0.001_seed1 res_n8_L64_s0.001_seed0 res_n8_L64_s0.001_seed1 res_n16_L256_s0.001_seed0 res_n16_L256_s0.001_seed1 res_n8_L256_s0.001_seed0 res_n8_L256_s0.001_seed1 plain_n16_L256_s0.001_seed0 plain_n16_L256_s0.001_seed1 plain_n16_L64_s0.001_seed0 plain_n16_L64_s0.001_seed1 plain_n8_L64_s0.001_seed0 plain_n8_L64_s0.001_seed2 plain_n8_L256_s0.001_seed2 plain_n8_L256_s0.001_seed7"
for pass in $(seq 1 80); do
  did=0
  for j in runs/p3/truth_*.json; do
    case $j in *_att.json|*_samplers.json|*_tails.json) continue;; esac
    p=${j%.json}
    [ -f ${p}_oracle.npz ] || continue
    if [ ! -f ${p}_att.json ]; then $N9 $PY scratch/p3_att.py $p >> logs/p3_att.log 2>&1; did=1; fi
  done
  for k in $SAMP; do
    p=runs/p3/truth_$k
    [ -f ${p}_oracle.npz ] || continue
    if [ ! -f ${p}_samplers.json ]; then $N9 $PY scratch/p3_samplers.py $p 16 >> logs/p3_samplers.log 2>&1; did=1; break; fi
  done
  if [ -f logs/q_truth_a.done ] && [ -f logs/q_truth_b.done ] && [ $did = 0 ]; then break; fi
  [ $did = 0 ] && sleep 60
done
echo DONE > logs/q_p3post.done
