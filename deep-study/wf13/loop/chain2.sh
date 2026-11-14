#!/bin/bash
# registered runs, second batch (three processes), started when the first batch has finished
cd /home/agent/workspaces/mats/work/deep-study/wf13/loop
PY="nice -n 19 ionice -c3 /home/agent/.venvs/aggext/bin/python -B"
until grep -q "batch1 done" logs/CHAIN_LOG 2>/dev/null; do sleep 5; done
$PY srun.py 2 > logs/chain_s2.out 2>&1 &
$PY prun.py 101 1 > logs/chain_p101.out 2>&1 &
$PY pscale.py ckpt/p_w103_s1.pt 10 > logs/chain_pscale.out 2>&1 &
wait
echo batch2 done $(date -u +%H:%M:%S) >> logs/CHAIN_LOG
