#!/bin/bash
# Dev sweep on census-healthy dev seeds (not evaluation nets), to calibrate the PREREG predictions.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/agent/workspaces/mats/work/unrestricted
PY=/home/agent/.venvs/arc-noisy/bin/python
for cfg in 3:8:901 4:8:901 4:32:907 6:8:900 6:32:901 8:8:900 8:32:900; do
  IFS=: read n L s <<< "$cfg"
  [ -f runs/dev/truth_n${n}_L${L}_seed${s}.json ] || $PY scratch/truth_baselines.py $n $L $s --mT 20 --RT 4 --mmax 18 --out runs/dev/truth_n${n}_L${L}_seed${s}.json
  for pr in gap instab; do
    $PY scratch/run_bnb.py $n $L $s --kmax 23 --max_seconds 240 --priority $pr --out runs/dev/bnb_${pr}_n${n}_L${L}_seed${s}.json
  done
done
echo DONE
