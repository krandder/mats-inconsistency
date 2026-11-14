#!/usr/bin/env bash
set -x
RID=$1; PY=/root/venv/bin/python; O=/root/runs/$RID/truth
nvidia-smi -L
S=0.001,0.003,0.01,0.03,0.1
for seed in 17 16 15 14 13 12 11 10; do $PY truth_gpu.py $O 26 $S $seed --m2 8,32; done
echo TRUTH3-DONE
