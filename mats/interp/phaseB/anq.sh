#!/usr/bin/env bash
# analysis queue (1 CPU slot): task-1 coarse-graining, then eval + MI for every finished noisy run
cd /home/agent/workspaces/mats/work/cm-interp-B
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/agent/.venvs/aggext/bin/python
[ -e results/task1_coarse.json ] || nice -n 5 $PY task1.py coarse > results/task1_coarse.log 2>&1
for pass in $(seq 1 200); do
  left=0
  for m in L2 L3; do for s in 0.0 0.05 0.1 0.2 0.4 0.8 1.6; do
    r=runs/${m}_s$s
    if [ ! -e $r/meta.json ]; then left=1; continue; fi
    [ -e $r/eval.json ] || nice -n 5 $PY nnoise.py eval --run $r > $r/eval.log 2>&1
    if [ $s != 0.0 ] && [ ! -e $r/mi.done ]; then nice -n 5 $PY mirun.py $r > $r/mi.log 2>&1 && touch $r/mi.done; fi
  done; done
  [ $left = 0 ] && break
  sleep 60
done
echo ANQ-DONE
