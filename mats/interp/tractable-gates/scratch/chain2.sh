#!/usr/bin/env bash
# Second K worker (reverse order, widest first) once the tanhb cut runs are done, keeping at most 3 heavy processes.
cd /home/agent/workspaces/mats/work/tractable-gates
until grep -q "cut_tanhb_n16 exit" logs/queue.done 2>/dev/null; do sleep 20; done
bash scratch/runq.sh scratch/queue_K_rev.txt 1
