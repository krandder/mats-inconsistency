#!/usr/bin/env bash
# run C2: E4 transcoder chains (fixed closure), arc sigma = 0.01, eval seed 65, n = 64 and 256
set -x
RID=$1; PY=/root/venv/bin/python; O=/root/runs/$RID
nvidia-smi -L
for n in 64 256; do
  $PY gpu/run_e4.py $O $n 0.01 65 || echo "FAILED e4 $n"
done
echo JOBC2-DONE
