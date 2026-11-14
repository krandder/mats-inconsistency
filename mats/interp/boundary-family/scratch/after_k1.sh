#!/bin/bash
cd /home/agent/workspaces/mats/work/boundary-family
while pgrep -f "kfrun.sh" > /dev/null; do sleep 20; done
TH=3 nice -n 19 ionice -c3 /home/agent/.venvs/arc-noisy/bin/python scratch/lcg256.py 101 2 4 > logs/lcg256_101_k2B4.log 2>&1
