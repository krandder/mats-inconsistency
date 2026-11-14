#!/usr/bin/env bash
# E5 on its own VM, packed as two parallel processes (coordinator: pack runs per VM)
set -x
RID=$1; PY=/root/venv/bin/python; O=/root/runs/$RID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -L
mkdir -p $O
$PY gpu/run_e5.py $O 256 0.01 65 --nfit 16 >> $O/e5_256.log 2>&1 &
P1=$!
( $PY gpu/run_e5.py $O 64 0.01 65; $PY gpu/run_e5.py $O 128 0.01 65; $PY gpu/run_e5.py $O 64 0.01 66 ) >> $O/e5_small.log 2>&1 &
P2=$!
wait $P1 $P2
tail -n 3 $O/e5_256.log $O/e5_small.log
echo JOBE5B-DONE
