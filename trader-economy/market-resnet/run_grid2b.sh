#!/bin/bash
# on trader-mresnet-5. Phase-1 additions, seeds 1-2 (moved off trader-mresnet-3 for wall-clock):
# Bansal P and the Ctt settlement variants; plus the train-time cost sweep of the fixed
# priced-halting arm (Ctt_eps at halt_cost 0.01 and 0.2; 0.05 is the default run), seeds 0-2.
# Then evaluate every checkpoint here.
cd ~ && mkdir -p results
{ for s in 1 2; do for arm in P Ctt_eps Ctt_real Ctt_exec; do echo "${arm}_s$s --arm $arm --seed $s"; done; done
  for c in 0.01 0.2; do for s in 0 1 2; do echo "Ctt_eps_c${c}_s$s --arm Ctt_eps --halt_cost $c --seed $s"; done; done
} | xargs -P 7 -L 1 bash -c 'n=$0; [ -f results/$n.pt ] || python3 train.py "$@" --iters 10000 --out results/$n.pt >> results/$n.log 2>&1'
python3 evaluate.py results/*_s?.pt > results/eval_grid2b.log 2>&1
echo GRID2B_DONE >> results/eval_grid2b.log
