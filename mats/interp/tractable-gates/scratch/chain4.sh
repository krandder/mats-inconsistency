#!/usr/bin/env bash
# Theory numbers left over (ray model): smoothstep1, relu2 and the GLU arms at vR = 1, after the tanhb cut runs.
cd /home/agent/workspaces/mats/work/tractable-gates
until grep -q "cut_tanhb_n8 exit" logs/queue.done 2>/dev/null; do sleep 20; done
RAY_TAG=_rest RAY_ARMS=smoothstep1,relu2,glu-silu,glu-gelu,glu-linear RAY_VR=1.0 bash scratch/runq.sh scratch/queue_ray.txt 1
