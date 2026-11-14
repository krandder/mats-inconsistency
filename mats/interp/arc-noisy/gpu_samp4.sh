#!/usr/bin/env bash
set -x
RID=$1; PY=/root/venv/bin/python; O=/root/runs/$RID/samp_v2
nvidia-smi -L
$PY -c "import scipy" 2>/dev/null || /root/venv/bin/pip install -q scipy
E=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23
$PY gpu_samplers.py $O 100,101,102,103,104,105,106,107 0.001 mub 12
$PY gpu_samplers.py $O $E 0.001,0.003 mub,sobol 12
$PY gpu_samplers.py $O $E 0.1,0.03,0.01 mub,mubanti 8
echo SAMP4-DONE
