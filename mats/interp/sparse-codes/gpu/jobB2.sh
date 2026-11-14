#!/usr/bin/env bash
# v2: tanh ordered / chaotic sigma = 0.01 (E1 mean origin, E2 incl. post-LN), then arc sigma = 0.001 n = 64, 256 (E1, E2)
set -x
RID=$1; PY=/root/venv/bin/python; O=/root/runs/$RID
nvidia-smi -L
for n in 64 128 256; do
  $PY gpu/run_main2.py $O tcha $n 0.01 65 || echo "FAILED tcha $n"
  $PY gpu/run_main2.py $O tord $n 0.01 65 || echo "FAILED tord $n"
done
for n in 64 256; do
  $PY gpu/run_main2.py $O arc $n 0.001 65 || echo "FAILED arc0.001 $n"
done
echo JOB-DONE
