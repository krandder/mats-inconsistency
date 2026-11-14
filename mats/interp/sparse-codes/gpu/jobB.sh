#!/usr/bin/env bash
# run B: tanh ordered / chaotic sigma = 0.01 (E1, E2), then arc sigma = 0.001 (E1, E2, E3), eval seed 65
set -x
RID=$1; PY=/root/venv/bin/python; O=/root/runs/$RID
nvidia-smi -L
for n in 64 128 256; do
  $PY gpu/run_main.py $O tcha $n 0.01 65 || echo "FAILED tcha $n"
  $PY gpu/run_main.py $O tord $n 0.01 65 || echo "FAILED tord $n"
done
for n in 64 256 128; do
  $PY gpu/run_main.py $O arc $n 0.001 65 --e3 4,8,16,24 || echo "FAILED arc0.001 $n"
done
echo JOBB-DONE
