#!/usr/bin/env bash
# Pinned sorting arms (Amendment 2) in the T slot: start when the T queue is done (keeps <= 3 heavy processes).
cd /home/agent/workspaces/mats/work/tractable-gates
until grep -q "QUEUE scratch/queue_T.txt ALL-DONE" logs/queue.done 2>/dev/null; do sleep 20; done
PIN=1 bash scratch/runq.sh scratch/queue_P.txt 1
