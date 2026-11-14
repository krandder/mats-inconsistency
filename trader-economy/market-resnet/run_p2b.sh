#!/bin/bash
# phase 2b (bounties): cov markets paid by traders instead of the house. Runs after run_p2.sh.
cd ~
until grep -q P2_DONE p2/p2.status 2>/dev/null; do sleep 60; done
C="--iters 3000 --bs 64 --D 24"
run() { n=$1; shift; [ -f p2/$n.done ] && return; L=""; [ -f p2/$n.pt ] && L="--load p2/$n"
        python3 econ2.py "$@" $C $L --out p2/$n >> p2/$n.log 2>&1 && touch p2/$n.done; }
export -f run; export C
for s in 0 1; do for b in vol lindahl; do
  echo "econ_cov_voc_bounty${b}_s$s --mode econ --stream cov --voc 1 --bounty $b --seed $s --evalL 32,320,3200"
done; done | xargs -P 4 -I{} bash -c 'run {}'
echo P2B_DONE >> p2/p2.status
