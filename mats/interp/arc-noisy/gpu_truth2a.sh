#!/usr/bin/env bash
set -x
RID=$1; PY=/root/venv/bin/python; O=/root/runs/$RID/truth
nvidia-smi -L
S=0.001,0.003,0.01,0.03,0.1
$PY truth_gpu.py $O 25 0.03,0.1 100 --m2 8,16,24,31,32
$PY truth_gpu.py $O 25 $S 101,102,103,104,105,106,107,108,109,110,111 --m2 8,16,24,31,32
$PY truth_gpu.py $O 30 0 200,201,202,203,204 --L 8 --rb ''
$PY truth_gpu.py $O 26 0 200,201,202,203,204 --L 32 --rb '' --m2 32
$PY truth_gpu.py $O 26 $S 18,19,20,21,22,23 --m2 8,32
echo TRUTH2A-DONE
