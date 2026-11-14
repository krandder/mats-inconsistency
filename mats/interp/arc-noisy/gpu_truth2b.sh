#!/usr/bin/env bash
set -x
RID=$1; PY=/root/venv/bin/python; O=/root/runs/$RID/truth
nvidia-smi -L
S=0.001,0.003,0.01,0.03,0.1
$PY truth_gpu.py $O 26 $S 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 --m2 8,32
echo TRUTH2B-DONE
