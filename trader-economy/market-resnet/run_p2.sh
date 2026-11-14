#!/bin/bash
# phase-2 grid: econ2 mechanisms vs BPTT, 3000 iters, D=24, lengths 8..32. Resumable:
# a finished run leaves p2/<name>.done; a trained but unevaluated run (.pt) is re-evaluated with --load.
cd ~ && mkdir -p p2 results
C="--iters 3000 --bs 64 --D 24"
run() { n=$1; shift; [ -f p2/$n.done ] && return; L=""; [ -f p2/$n.pt ] && L="--load p2/$n"
        python3 econ2.py "$@" $C $L --out p2/$n >> p2/$n.log 2>&1 && touch p2/$n.done; }
export -f run; export C
{
for s in 0 1 2; do echo "econ_cov_voc_s$s --mode econ --stream cov --voc 1 --seed $s --evalL 32,320,3200,32000"; done
for s in 0 1 2; do echo "bptt_cov_s$s --mode bptt --stream cov --seed $s --evalL 32,320,3200,32000"; done
for s in 0 1; do
  echo "econ_none_s$s --mode econ --stream none --seed $s --evalL 32,320,3200"
  echo "bptt_cov_prog_s$s --mode bptt --stream cov --prog 1 --seed $s --evalL 32,320,3200,32000"
  echo "econ_cov_voc_nocap_s$s --mode econ --stream cov --voc 1 --wealth_cap 0 --seed $s --evalL 32,320,3200"
  echo "econ_cov_voc_haltreal_s$s --mode econ --stream cov --voc 1 --train_halt 0.002 --settle real --seed $s --evalL 32,320"
  echo "econ_cov_voc_halteps_s$s --mode econ --stream cov --voc 1 --train_halt 0.002 --settle eps --eps 0.1 --seed $s --evalL 32,320"
  echo "econ_slots_s$s --mode econ --stream slots --seed $s --evalL 32,320,3200"
  echo "bptt_none_s$s --mode bptt --stream none --seed $s --evalL 32,320,3200"
  echo "econ_cov_voc_grow_s$s --mode econ --stream cov --voc 1 --grow_every 500 --seed $s --evalL 32,320,3200"
done
} | xargs -P 4 -I{} bash -c 'run {}'
[ -f p2/memdepth.done ] || { python3 memdepth.py > p2/memdepth.log 2>&1; cp results/memdepth.json p2/ && touch p2/memdepth.done; }
echo P2_DONE >> p2/p2.status
