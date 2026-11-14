#!/bin/bash
# stage2.sh (checker, post hoc, beyond the registered scale): the same world-103 pipeline at K = 1, 4, 16.
# Starts only when the stage-1 training shards have finished, so that at most 3 processes run.
cd /home/agent/workspaces/mats/work/deep-study/wf13/consistency/check
PY=/home/agent/.venvs/aggext/bin/python
until ! pgrep -u kas -f "[t]r.py data/w103 out/w103 jobs1.txt" > /dev/null; do sleep 3; done
date -u
G="nice -n 19 ionice -c3 $PY -B gen.py 103"
$G lab2 16 data/w103/lab_K16.pt 512 > logs/gen2_lab.log 2>&1 &
$G t2 32 data/w103/t2_K32.pt 96 > logs/gen2_t2.log 2>&1 &
( $G t1 16 data/w103/t1_K16.pt 256 > logs/gen2_t1.log 2>&1; $G c1 16 data/w103/c1_K16.pt 128 > logs/gen2_c1.log 2>&1 ) &
wait
date -u
for s in 0 1 2; do SUF=_K16 LEVELS=1,4,16 nice -n 19 ionice -c3 $PY -B tr.py data/w103 out/w103_K16 jobs2.txt $s 3 > logs/tr2_$s.log 2>&1 & done
wait
date -u
echo stage2 done
