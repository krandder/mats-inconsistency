#!/bin/bash
# POST HOC extension (coordinator 00:00Z: r(eps) scaling headline): start queue2 when queue1 has finished.
cd /home/agent/workspaces/mats/work/cut-rank
until grep -q "queue scratch/queue1.txt finished" logs/queue.done 2>/dev/null; do sleep 30; done
./scratch/run_queue.sh scratch/queue2.txt 3
