#!/bin/sh
# Part S queue (after Amendment S1): waits for the grid test (pid 4008234), then the registered runs.
cd /home/agent/workspaces/mats/work/stationary-numerics
while kill -0 4008234 2>/dev/null; do sleep 10; done
for s in 1000 1001 1002 1003 1004 1005; do OMP_NUM_THREADS=1 nice -n 19 ionice -c3 /home/agent/.venvs/arc-noisy/bin/python scratch/smalln_density.py 2 $s 40 0.1 0.05 --out runs/S_n2_s$s.json --save-grid runs/S_n2_s$s.npz; done
for s in 1000 1001 1002 1003; do OMP_NUM_THREADS=1 nice -n 19 ionice -c3 /home/agent/.venvs/arc-noisy/bin/python scratch/smalln_density.py 3 $s 30 0.2 0.14 --out runs/S_n3_s$s.json; done
for s in 1000 1001; do OMP_NUM_THREADS=1 nice -n 19 ionice -c3 /home/agent/.venvs/arc-noisy/bin/python scratch/smalln_density.py 4 $s 16 0.5 0.4 --marg 6 --skip-two-starts --out runs/S_n4_s$s.json; done
