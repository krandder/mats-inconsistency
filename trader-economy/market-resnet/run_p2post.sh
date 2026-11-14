#!/bin/bash
# on trader-mresnet-4 after run_p2.sh:
#  1. bptt with deep supervision (loss on every step's belief), 3 seeds: the backprop arm with the
#     same per-step signal the economy is paid on (bptt_cov uses only the final step's loss)
#  2. bptt_cov_s0's L=32000 eval ran out of GPU memory while 4 jobs shared the GPU; redo its evals
#     alone from the saved checkpoint (no retraining)
cd ~
until grep -q P2_DONE p2/p2.status 2>/dev/null; do sleep 60; done
C="--iters 3000 --bs 64 --D 24"
run() { n=$1; shift; [ -f p2/$n.done ] && return; L=""; [ -f p2/$n.pt ] && L="--load p2/$n"
        python3 econ2.py "$@" $C $L --out p2/$n >> p2/$n.log 2>&1 && touch p2/$n.done; }
export -f run; export C
for s in 0 1 2; do echo "bptt_cov_ds_s$s --mode bptt --stream cov --ds 1 --seed $s --evalL 32,320,3200,32000"; done | xargs -P 3 -I{} bash -c 'run {}'
for s in 0 1 2; do run bptt_cov_ds_s$s --mode bptt --stream cov --ds 1 --seed $s --evalL 32,320,3200,32000; done  # retry alone if one ran out of memory
run bptt_cov_s0 --mode bptt --stream cov --seed 0 --evalL 32,320,3200,32000
echo P2POST_DONE >> p2/p2.status
