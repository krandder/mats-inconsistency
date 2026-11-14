#!/bin/bash
# M9(c) w3: 42 CPU runs of run_kq.py on the VM's 8 vCPUs (7 parallel); CUDA hidden so runs match CPU harness.
export CUDA_VISIBLE_DEVICES="" KQ_OUT=$OUT KQ_TMP=$OUT/tmp
mkdir -p $OUT/tmp; nproc; python -c "import torch;print(torch.__version__)"
cd study/M9/w3
for s in 1 2 3; do for arm in B A; do for kh in "16 8" "1 8" "4 8" "64 8" "1 128" "4 32" "64 2"; do
  echo "$arm $kh $s"; done; done; done | xargs -P 7 -L 1 python run_kq.py
ls $OUT | wc -l
