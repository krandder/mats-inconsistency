#!/usr/bin/env bash
# E5 (Amendment 2): deduction vs sampling on the E1 trees (W0, N = 12, seed 0, m = 2 and m = 16) and the dense tok
# net (sampling only). Three lanes at lowest priority. Usage: e5run.sh RUNDIR (the synced GPU run directory).
R=$1
cd /home/agent/workspaces/mats/work/scale-free-exp
export OMP_NUM_THREADS=1
P="nice -n 19 ionice -c3 /home/agent/.venvs/aggext/bin/python"
O=results/e5.jsonl
mkdir -p results
T2=$R/e1w0_tree2_N12_s0; T16=$R/e1w0_tree16_N12_s0
(
  $P est5.py $T2 12 mc --M 8000000 --seed 1 --out $O
  $P est5.py $T2 12 mc --M 1000000 --seed 2 --out $O
) > e5_lane1.log 2>&1 &
(
  $P est5.py $T2 12 exact --nsrc 13 --ny 10 --G 1200 --out $O
  $P est5.py $T2 12 exact --nsrc 14 --ny 10 --G 1600 --out $O
  $P est5.py $T2 12 gauss --nsrc 13 --ny 10 --out $O
) > e5_lane2.log 2>&1 &
(
  $P est5.py $T16 12 mc --M 8000000 --seed 1 --out $O
  $P est5.py $T16 12 gauss --nsrc 13 --ny 10 --out $O
  $P est5.py none 12 world --M 200000 --out $O
) > e5_lane3.log 2>&1 &
wait
echo E5-DONE
