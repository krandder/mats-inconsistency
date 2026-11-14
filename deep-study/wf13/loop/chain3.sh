#!/bin/bash
# post hoc chain, one process at a time (addendum A3)
cd /home/agent/workspaces/mats/work/deep-study/wf13/loop
PY="nice -n 19 ionice -c3 /home/agent/.venvs/aggext/bin/python -B"
$PY sdebug.py avoid > logs/chain_sdebug_avoid.out 2>&1
$PY sdiag.py 1 > logs/chain_sdiag1.out 2>&1
$PY srun.py 1 order=confmask tag=_mask arms=min n_mixed=60 n_multi=20 > logs/chain_mask1.out 2>&1
$PY pscale2.py 60 5 > logs/chain_pscale2_60.out 2>&1
echo chain3 done $(date -u +%H:%M:%S) >> logs/CHAIN_LOG
