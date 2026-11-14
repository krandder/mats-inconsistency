#!/bin/bash
# provenance reruns (claims review finding 12): width-64 deductive results to log files
cd /home/agent/workspaces/mats/work/boundary-family
PY="nice -n 19 ionice -c3 /home/agent/.venvs/arc-noisy/bin/python"
T=runs/explore1/truth_500_L16_N22.pt
TH=2 $PY scratch/lrg_test.py 500 16 $T > logs/w64_lrg.log 2>&1
TH=2 $PY scratch/lcg_test.py 500 16 $T 0:1:1 1:4:1 2:4:1 3:4:1 > logs/w64_lcg.log 2>&1
TH=2 $PY scratch/gm_test.py 500 16 $T none:1 latent:9 latent:27 latent:81 boundary:4 boundary:16 boundary:64 > logs/w64_gm.log 2>&1
TH=2 $PY scratch/grid_test.py 500 16 $T 0:1 1:4 2:4 3:4 > logs/w64_grid.log 2>&1
