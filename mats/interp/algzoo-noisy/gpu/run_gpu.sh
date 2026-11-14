#!/usr/bin/env bash
# Runs on the L4 VM from /root/runs/$RID: three batched from-scratch trainings (ARC's recipe, 2^30 sequences).
set -u
PY=${PY:-/root/venv/bin/python}
mkdir -p out
nvidia-smi -L
$PY train_batched.py 16 10 "0:0,1:0,0:0.003,1:0.003,0:0.01,1:0.01" ${NTRAIN:-1073741824} out/w16 cuda > out/w16.log 2>&1 &
$PY train_batched.py 32 10 "0:0.01,1:0.01,2:0.01,0:0.03,1:0.03,2:0.03" ${NTRAIN:-1073741824} out/w32 cuda > out/w32.log 2>&1 &
$PY train_batched.py 64 10 "0:0.01,1:0.01,2:0.01,0:0.03,1:0.03,2:0.03" ${NTRAIN:-1073741824} out/w64 cuda > out/w64.log 2>&1 &
wait
tail -3 out/*.log
echo RUN-GPU-DONE
