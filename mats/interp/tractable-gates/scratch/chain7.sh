#!/usr/bin/env bash
# T slot after the sorting queue: grid R on the trained sorting nets, then the pinned arms (Amendment 2), then grid R on them.
cd /home/agent/workspaces/mats/work/tractable-gates
until grep -q "QUEUE scratch/queue_T.txt ALL-DONE" logs/queue.done 2>/dev/null; do sleep 20; done
bash scratch/runq.sh scratch/queue_R.txt 1
PIN=1 bash scratch/runq.sh scratch/queue_P.txt 1
bash scratch/runq.sh scratch/queue_Rpin.txt 1
