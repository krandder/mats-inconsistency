#!/bin/bash
# K1' (PREREG.md): nets 101 and 102, cut 24, families G, PCA3, AS3, GATE6 (representation + push)
cd /home/agent/workspaces/mats/work/boundary-family
for s in 101 102; do
  TH=3 nice -n 19 ionice -c3 /home/agent/.venvs/arc-noisy/bin/python scratch/diag5.py $s 24 18 20 G,PCA3,AS3,GATE6 > logs/k1_${s}_m24.log 2>&1
done
