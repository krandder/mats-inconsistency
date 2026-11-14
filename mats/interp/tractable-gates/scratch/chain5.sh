#!/usr/bin/env bash
# Replaces chain2/chain4: after the tanhb cut runs, the leftover ray-model arms (one worker), then a second K worker in
# reverse order. Keeps at most 3 heavy processes (K forward worker, T, and this one). Waits poll logs/queue.done only.
cd /home/agent/workspaces/mats/work/tractable-gates
until grep -q "cut_tanhb_n16 exit" logs/queue.done 2>/dev/null; do sleep 20; done
RAY_TAG=_rest RAY_ARMS=smoothstep1,relu2,glu-silu,glu-gelu,glu-linear RAY_VR=1.0 bash scratch/runq.sh scratch/queue_ray.txt 1
bash scratch/runq.sh scratch/queue_K_rev.txt 1
