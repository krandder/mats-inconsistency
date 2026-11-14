#!/usr/bin/env bash
# resume paused trainings one at a time, in priority order, whenever the lane has a free slot
cd /home/agent/workspaces/mats/work/cm-interp-B
active() { ps -eo stat,args | grep -E "python[^ ]* (-W ignore )?(nnoise|mirun|task1|task3|e1|e2|e3|pfn_rare|pfn_diag|pfn_is|dec|twist)\.py" | grep -v grep | grep -v "bash " | awk '$1 !~ /T/' | wc -l; }
for p in 1648172 1872104 1763239 1715969 1494111 1458513; do
  while [ $(active) -ge 4 ]; do sleep 41; done
  kill -CONT $p; sleep 30
done
echo RESUMER-DONE
