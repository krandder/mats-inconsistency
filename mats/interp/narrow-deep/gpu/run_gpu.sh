#!/usr/bin/env bash
# Runs on the L4 VM from /root/runs/$RID: narrow-deep trainings (PREREG Part 1, Part 4), all concurrently.
set -u
PY=${PY:-/root/venv/bin/python}
mkdir -p out
nvidia-smi -L
export LOG_EVERY=1024
BATCH=4096 WARM_STEPS=4096 $PY rnn_train_gpu.py 16 64 "0:0,1:0,2:0,3:0,0:0.01,1:0.01,2:0.01,3:0.01" 65536 out/rnn_T64 cuda > out/rnn_T64.log 2>&1 &
BATCH=4096 WARM_STEPS=4096 $PY rnn_train_gpu.py 16 32 "0:0,1:0,0:0.01,1:0.01" 65536 out/rnn_T32 cuda > out/rnn_T32.log 2>&1 &
$PY mlp_train.py res 16 16 "0:0,1:0,0:0.01,1:0.01" 32768 out/mlp cuda > out/mlp_res16.log 2>&1 &
$PY mlp_train.py res 16 64 "0:0,1:0,0:0.01,1:0.01" 32768 out/mlp cuda > out/mlp_res64.log 2>&1 &
$PY mlp_train.py res 16 128 "0:0,1:0,0:0.01,1:0.01" 32768 out/mlp cuda > out/mlp_res128.log 2>&1 &
$PY mlp_train.py plain 16 16 "0:0,1:0,0:0.01" 32768 out/mlp cuda > out/mlp_plain16.log 2>&1 &
$PY mlp_train.py plain 16 64 "0:0,1:0,0:0.01" 32768 out/mlp cuda > out/mlp_plain64.log 2>&1 &
wait
tail -4 out/*.log
echo RUN-GPU-DONE
