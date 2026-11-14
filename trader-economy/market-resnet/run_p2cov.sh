#!/bin/bash
# on trader-mresnet-8 after run_p2post8.sh: the BPTT control with every training signal the economy gets.
# bptt_cov_dscov = end to end through depth on every step's log loss (as bptt_cov_ds) plus the
# covariance targets on c at every step (the economy's covariance markets pay on these targets;
# bptt_cov and bptt_cov_ds leave c a free latent). 3 seeds.
cd ~
until grep -q P2POST_DONE p2/p2.status 2>/dev/null; do sleep 30; done
C="--iters 3000 --bs 64 --D 24"
run() { n=$1; shift; [ -f p2/$n.done ] && return; L=""; [ -f p2/$n.pt ] && L="--load p2/$n"
        python3 econ2.py "$@" $C $L --out p2/$n >> p2/$n.log 2>&1 && touch p2/$n.done; }
export -f run; export C
for s in 0 1 2; do echo "bptt_cov_dscov_s$s --mode bptt --stream cov --ds 1 --covsup 1 --seed $s --evalL 32,320,3200,32000"; done | xargs -P 3 -I{} bash -c 'run {}'
for s in 0 1 2; do run bptt_cov_dscov_s$s --mode bptt --stream cov --ds 1 --covsup 1 --seed $s --evalL 32,320,3200,32000; done  # retry alone if one ran out of memory
echo P2COV_DONE >> p2/p2.status
