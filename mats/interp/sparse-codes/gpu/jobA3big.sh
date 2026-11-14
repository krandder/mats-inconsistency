#!/usr/bin/env bash
# v2: arc sigma = 0.01, seed 65, n = 256 (E1 both origins, controls at 8/24, E2, E3)
set -x
RID=$1; PY=/root/venv/bin/python; O=/root/runs/$RID
nvidia-smi -L
$PY gpu/run_main2.py $O arc 256 0.01 65 --e3 4,8,16,24 --controls 8,24 --save-sae 8,16,24 || echo "FAILED arc 256"
echo JOB-DONE
