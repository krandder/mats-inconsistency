#!/bin/bash
# phase-1 grid on the GPU VM: 6 arms x 3 seeds (+ PonderNet prior variant), 10k iters, 5 in parallel
cd ~
jobs=""
for s in 0 1 2; do for arm in A B Bds C Ctt D; do jobs="$jobs $arm:$s:0"; done; jobs="$jobs D:$s:0.2"; done
echo $jobs | tr ' ' '\n' | xargs -P 5 -I{} bash -c 'IFS=: read arm s lp <<< "{}"; n=${arm}_s${s}; [ "$lp" != 0 ] && n=D_lp${lp}_s${s}; [ -f results/$n.pt ] && exit 0; python3 train.py --arm $arm --seed $s --iters 10000 --lam_prior $lp --out results/$n.pt > results/$n.log 2>&1'
python3 evaluate.py results/[A-D]*_s?.pt results/D_lp*.pt --credit > results/eval_all.log 2>&1
echo GRID_DONE >> results/eval_all.log
