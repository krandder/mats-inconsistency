#!/usr/bin/env bash
# Usage: launch.sh <run-id> [STEPS] [P]
set -euo pipefail
RID=$1; STEPS=${2:-40000}; P=${3:-6}
cd /home/agent/workspaces/mats/work/pfn-dedup
B=$(tar cz world.py pfn.py test_world.py summarize.py run_all.sh PREREG.md | base64 -w0)
CMD="mkdir -p /root/runs/$RID && cd /root/runs/$RID && echo $B | base64 -d | tar xz && nvidia-smi -L && /root/venv/bin/pip install -q scipy && export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore OUT=/root/runs/$RID/runs STEPS=$STEPS P=$P ARMS=${ARMS:-} PFN_WORLD=${PFN_WORLD:-} && bash run_all.sh"
GPU_PROVISIONING=${GPU_PROVISIONING:-SPOT} GPU_RUN_TIMEOUT=${GPU_RUN_TIMEOUT:-12600s} /home/agent/workspaces/mats/work/run_on_l4.sh "$RID" 0a36abb "$CMD"
