#!/usr/bin/env bash
# Chain the queues so that at most 3 heavy processes run: C (2 workers) and K (1 worker) are running; when C is done,
# start C2 (tanhb cut runs, 1 worker) and then T (sorting training, 1 worker); when K is done, start K2 (tanhb K runs).
# Waits poll logs/queue.done (a file), never the process list, so they cannot match themselves.
cd /home/agent/workspaces/mats/work/tractable-gates
until grep -q "QUEUE scratch/queue_C.txt ALL-DONE" logs/queue.done 2>/dev/null; do sleep 20; done
bash scratch/runq.sh scratch/queue_C2.txt 1 &
bash scratch/runq.sh scratch/queue_T.txt 1 &
until grep -q "QUEUE scratch/queue_K.txt ALL-DONE" logs/queue.done 2>/dev/null; do sleep 20; done
bash scratch/runq.sh scratch/queue_K2.txt 1 &
wait
echo "$(date -u +%H:%M:%SZ) CHAIN ALL-DONE" >> logs/queue.done
