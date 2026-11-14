#!/bin/bash
# phase-1 additions: Bansal et al. 2022 progressive loss (P) and the decision-market
# settlement variants of train-time priced halting (Ctt_real, Ctt_exec buggy; Ctt_eps fixed)
cd ~
jobs=""
for s in 0 1 2; do for arm in P Ctt_eps Ctt_real Ctt_exec; do jobs="$jobs $arm:$s"; done; done
echo $jobs | tr ' ' '\n' | xargs -P 2 -I{} bash -c 'a=$(echo {} | cut -d: -f1); s=$(echo {} | cut -d: -f2); [ -f results/${a}_s$s.pt ] || python3 train.py --arm $a --seed $s --iters 10000 --out results/${a}_s$s.pt > results/${a}_s$s.log 2>&1'
python3 evaluate.py results/P_s?.pt results/Ctt_eps_s?.pt results/Ctt_real_s?.pt results/Ctt_exec_s?.pt > results/eval_grid2.log 2>&1
echo GRID2_DONE >> results/eval_grid2.log
