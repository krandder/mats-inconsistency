#!/bin/bash
# on trader-mresnet-9, after -8 was preempted: finish run_p2cov.sh. dscov s0-1 have checkpoints (their
# evals rerun from them, no retraining); s2 was lost in its last training iterations and is retrained.
cd ~
C="--iters 3000 --bs 64 --D 24"
run() { n=$1; shift; [ -f p2/$n.done ] && return; L=""; [ -f p2/$n.pt ] && L="--load p2/$n"
        python3 econ2.py "$@" $C $L --out p2/$n >> p2/$n.log 2>&1 && touch p2/$n.done; }
export -f run; export C
for s in 0 1 2; do echo "bptt_cov_dscov_s$s --mode bptt --stream cov --ds 1 --covsup 1 --seed $s --evalL 32,320,3200,32000"; done | xargs -P 3 -I{} bash -c 'run {}'
for s in 0 1 2; do run bptt_cov_dscov_s$s --mode bptt --stream cov --ds 1 --covsup 1 --seed $s --evalL 32,320,3200,32000; done  # retry alone if one ran out of memory
echo P2COV_DONE >> p2/p2.status
