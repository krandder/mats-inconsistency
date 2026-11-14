#!/usr/bin/env bash
# M1 interaction profiles (Amendment 3) on the E1 nets (W0): set, setnc, tok at N = 3, 12 (seeds 0, 1), tree16 at
# N = 3, 12 (seed 0), tree2 N = 12; plus Monte Carlo speed of the tok net for E5's dense contrast.
# Usage: m1run.sh DIR_B3 DIR_B4 (directories holding the run folders)
B3=$1; B4=$2
cd /home/agent/workspaces/mats/work/scale-free-exp
export OMP_NUM_THREADS=1
P="nice -n 19 ionice -c3 /home/agent/.venvs/aggext/bin/python"
L1=""; L2=""
for N in 3 12; do
  for s in 0 1; do
    L1="$L1 $B3/e1w0_set_N${N}_s$s $B3/e1w0_setnc_N${N}_s$s"
    L2="$L2 $B4/e1w0_tok_N${N}_s$s"
  done
  L2="$L2 $B3/e1w0_tree16_N${N}_s0"
done
$P m1prof.py $L1 --n 3000 --out results/m1.jsonl > m1_lane1.log 2>&1 &
$P m1prof.py $L2 --n 3000 --out results/m1.jsonl > m1_lane2.log 2>&1 &
wait
echo M1-DONE
