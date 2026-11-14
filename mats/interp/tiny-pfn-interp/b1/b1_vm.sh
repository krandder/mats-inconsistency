#!/usr/bin/env bash
# B1 on the VM: test set, 5 fresh seeds (Gaussian 160k -> warm-started mixture), eval, locate + steps.
set -uo pipefail
R=$(pwd); PY=/root/venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore TINYPFN_DIR=$R/tp
cd $R/tp && mkdir -p runs && $PY pfn.py testset --out runs
$PY -c "
import numpy as np, hashlib
T=np.load('runs/testset.npz'); h=lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
print('TESTSET-HASH', h(T['X']), h(T['w_bayes']))"
nproc
for S in 5 6 7 8 9; do (
  $PY pfn.py train --width 32 --heads 2 --depth 2 --phase1_fixed 160000 --seed $S --out runs/w32_h2_L2_long_s$S > runs/w32_h2_L2_long_s$S.log 2>&1
  $PY pfn.py train --width 32 --heads 2 --depth 2 --mixk 4 --seed $S --init_from runs/w32_h2_L2_long_s$S/model.pt --out runs/w32_h2_L2_mix4_ws_s$S > runs/w32_h2_L2_mix4_ws_s$S.log 2>&1
  $PY pfn.py eval --run runs/w32_h2_L2_long_s$S --testset runs/testset.npz | cut -c1-200
  $PY pfn.py eval --run runs/w32_h2_L2_mix4_ws_s$S --testset runs/testset.npz | cut -c1-200
  cd $R/interp && $PY locate.py runs/w32_h2_L2_mix4_ws_s$S main_s$S out/main_s$S.json > out/main_s$S.log 2>&1
  $PY steps.py runs/w32_h2_L2_mix4_ws_s$S main_s$S out/steps_main_s$S.json > out/steps_main_s$S.log 2>&1
  echo "SEED-DONE $S $(date -u +%H:%M)"
) & done
wait
rm -f $R/tp/runs/testset.npz
echo B1-COMPLETE
