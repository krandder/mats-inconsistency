#!/bin/sh
# Part D queue: waits for the Gram pr run (pid 3845462) to end, then runs the registered response runs.
cd /home/agent/workspaces/mats/work/stationary-numerics
while kill -0 3845462 2>/dev/null; do sleep 10; done
for n in 8 16 32 64 128 256; do case $n in 128|256) P=64;; *) P=128;; esac; for s in 1000 1001 1002 1003; do OMP_NUM_THREADS=1 nice -n 19 ionice -c3 /home/agent/.venvs/arc-noisy/bin/python scratch/response_net.py $n $s --P $P --out runs/D_n${n}_s$s.json; done; done
