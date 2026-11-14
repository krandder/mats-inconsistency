#!/usr/bin/env bash
# Amendment 5 on the VM: test set, 25 runs (5 arms x seeds 10-14) on the GPU, 8 workers, eval + budget_lite per run.
set -uo pipefail
R=$(pwd); PY=/root/venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore TINYPFN_DIR=$R/tp R PY
nvidia-smi -L
cd $R/tp && mkdir -p runs && $PY pfn.py testset --out runs
$PY -c "
import numpy as np, hashlib
T=np.load('runs/testset.npz'); h=lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
print('TESTSET-HASH', h(T['X']), h(T['w_bayes']))"
mkdir -p $R/hruns $R/hout
job() {
  arm=$1; S=$2
  case $arm in base) A="";; h3) A="--l2_heads 3 --l2_dh 16";; h4) A="--l2_heads 4 --l2_dh 16";; w64) A="--l2_heads 2 --l2_dh 32";; L3) A="--depth 3";; esac
  G=$R/hruns/${arm}_g_s$S; M=$R/hruns/${arm}_ws_s$S
  cd $R/interp/heads
  t0=$(date +%s)
  $PY pfn2.py train --width 32 --heads 2 --depth 2 $A --phase1_fixed 160000 --seed $S --out $G > $G.log 2>&1
  $PY pfn2.py train --width 32 --heads 2 --depth 2 $A --mixk 4 --seed $S --init_from $G/model.pt --out $M > $M.log 2>&1
  $PY pfn2.py eval --run $M --testset $R/tp/runs/testset.npz > $M.eval.txt 2>&1
  $PY budget_lite.py $M ${arm}_s$S $R/hout/${arm}_s$S.json > $R/hout/${arm}_s$S.log 2>&1
  echo "JOB-DONE $arm $S $(( $(date +%s) - t0 ))s $(date -u +%H:%M) $(cut -c1-120 $M.eval.txt)"
}
export -f job
for S in 10 11 12 13 14; do for a in base h4 h3 w64 L3; do echo "$a $S"; done; done | xargs -P 8 -L 1 bash -c 'job $0 $1'
rm -f $R/tp/runs/testset.npz
echo HEADS-COMPLETE
