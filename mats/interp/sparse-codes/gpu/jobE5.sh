#!/usr/bin/env bash
# E5: feature copula vs exact latent (PREREG amendment 4)
set -x
RID=$1; PY=/root/venv/bin/python; O=/root/runs/$RID
nvidia-smi -L
$PY gpu/run_e5.py $O 64 0.01 65 || echo "FAILED e5 64"
$PY gpu/run_e5.py $O 256 0.01 65 || echo "FAILED e5 256"
$PY gpu/run_e5.py $O 128 0.01 65 || echo "FAILED e5 128"
$PY gpu/run_e5.py $O 64 0.01 66 || echo "FAILED e5 64 seed66"
echo JOBE5-DONE
