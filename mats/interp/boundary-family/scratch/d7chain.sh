#!/bin/bash
# measurement D (PREREG): width trend at L/n = 1/8 (64x8, 128x16) and net 102 at width 256
cd /home/agent/workspaces/mats/work/boundary-family
WID=64 LNET=8 TH=3 nice -n 19 ionice -c3 /home/agent/.venvs/arc-noisy/bin/python scratch/diag7w.py 500 4,6,7 18 > logs/diag7_500_w64.log 2>&1
WID=128 LNET=16 TH=3 nice -n 19 ionice -c3 /home/agent/.venvs/arc-noisy/bin/python scratch/diag7w.py 600 8,12,15 17 > logs/diag7_600_w128.log 2>&1
TH=3 nice -n 19 ionice -c3 /home/agent/.venvs/arc-noisy/bin/python scratch/diag7.py 102 16,24,31 16 > logs/diag7_102.log 2>&1
