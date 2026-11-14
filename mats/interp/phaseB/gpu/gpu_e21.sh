#!/usr/bin/env bash
# E21 job on one L4 VM: train the tiny-pfn recipe (2 layers, 2 heads, mixture head K = 4) at widths 64, 128, 256, seeds 0-4,
# with pfn2.py (the CUDA copy of pfn.py used by the heads cohort). Up to 6 trainings at once. Outputs: /root/runs/$RID/w{W}_s{S}/
set -x
RID=$1; PY=/root/venv/bin/python; OUT=/root/runs/$RID
nvidia-smi -L
mkdir -p $OUT
export OMP_NUM_THREADS=1
jobs_running() { jobs -rp | wc -l; }
for W in 64 128 256; do
  for S in 0 1 2 3 4; do
    while [ "$(jobs_running)" -ge 6 ]; do sleep 20; done
    $PY pfn2.py train --width $W --heads 2 --depth 2 --mixk 4 --seed $S --out $OUT/w${W}_s${S} > $OUT/w${W}_s${S}.log 2>&1 &
    sleep 5
  done
done
wait
ls $OUT
echo E21-DONE
