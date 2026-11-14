#!/usr/bin/env bash
# run A: arc sigma = 0.01, eval seed 65, widths 64/128/256 (E1, E2, E3)
set -x
RID=$1; PY=/root/venv/bin/python; O=/root/runs/$RID
nvidia-smi -L
for n in 64 128 256; do
  $PY gpu/run_main.py $O arc $n 0.01 65 --e3 4,8,16,24 --save-sae 8,16,24 --long-control 8,24 || echo "FAILED arc $n"
done
echo JOBA-DONE
