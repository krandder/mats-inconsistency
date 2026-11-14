#!/usr/bin/env bash
set -x
RID=$1; PY=/root/venv/bin/python; O=/root/runs/$RID/sampanti
nvidia-smi -L
E=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23
$PY gpu_samplers.py $O $E 0.1,0.03,0.01 mubanti 8
echo SAMP3-DONE
