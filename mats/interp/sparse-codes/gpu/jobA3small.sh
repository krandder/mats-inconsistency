#!/usr/bin/env bash
# v2: arc sigma = 0.01, seed 65, n = 64 and 128; then the seed-66 replicate at n = 64
set -x
RID=$1; PY=/root/venv/bin/python; O=/root/runs/$RID
nvidia-smi -L
$PY gpu/run_main2.py $O arc 64 0.01 65 --e3 4,8,16,24 --controls 8,24 --save-sae 8,16,24 || echo "FAILED arc 64"
$PY gpu/run_main2.py $O arc 128 0.01 65 --e3 4,8,16,24 --controls 8,24 --save-sae 8,16,24 || echo "FAILED arc 128"
$PY gpu/run_main2.py $O arc 64 0.01 66 --e3 4,8,16,24 --controls 8,24 || echo "FAILED arc 64 seed66"
echo JOB-DONE
