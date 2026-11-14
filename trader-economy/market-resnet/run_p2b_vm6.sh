#!/bin/bash
# on trader-mresnet-6. Phase 2b (bounties): cov markets paid by traders instead of the house.
# Plus priced halting with a nonzero compute cost (never tested before: every earlier run used
# --cost 0): cost c charged per executed step, VOC halting at c during training with
# eps-continuation (0.1) and 1/eps settlement, c in {1e-3, 3e-3, 1e-2}, seeds 0-1.
cd ~ && mkdir -p p2
C="--iters 3000 --bs 64 --D 24"
run() { n=$1; shift; [ -f p2/$n.done ] && return; L=""; [ -f p2/$n.pt ] && L="--load p2/$n"
        python3 econ2.py "$@" $C $L --out p2/$n >> p2/$n.log 2>&1 && touch p2/$n.done; }
export -f run; export C
{ for s in 0 1; do for b in vol lindahl; do
    echo "econ_cov_voc_bounty${b}_s$s --mode econ --stream cov --voc 1 --bounty $b --seed $s --evalL 32,320,3200"
  done; done
  for c in 1e-3 3e-3 1e-2; do for s in 0 1; do
    echo "econ_cov_voc_priced_c${c}_s$s --mode econ --stream cov --voc 1 --cost $c --train_halt $c --settle eps --eps 0.1 --seed $s --evalL 32,320"
  done; done
} | xargs -P 5 -I{} bash -c 'run {}'
echo P2B_DONE >> p2/p2.status
