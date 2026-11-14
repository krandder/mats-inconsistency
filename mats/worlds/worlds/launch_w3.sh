#!/usr/bin/env bash
# Usage: launch_w3.sh <run-id>   (env passes through: STEPS, P, SEEDS, ARMS)
set -euo pipefail
RID=$1
cd /home/agent/workspaces/mats/work/worlds
B=$(tar cz harness.py looped.py world3_chains.py test_world3_chains.py test_looped.py w3_summarize.py w3_dense.py run_w3.sh run_w3_screen.sh PREREG-W3.md | base64 -w0)
CMD="mkdir -p /root/runs/$RID && cd /root/runs/$RID && echo $B | base64 -d | tar xz && nvidia-smi -L && export PATH=/root/venv/bin:\$PATH OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore OUT=/root/runs/$RID/runs STEPS=${STEPS:-40000} P=${P:-5} SEEDS='${SEEDS:-0 1 2}' ARMS='${ARMS:-U2 U4 U6 LP2x2 LP2x3}' WCFG='${WCFG:-{\}}' && SCREEN=${SCREEN:-1} bash ${SCRIPT:-run_w3.sh}"
GPU_PROVISIONING=${GPU_PROVISIONING:-STANDARD} GPU_RUN_TIMEOUT=${GPU_RUN_TIMEOUT:-12600s} /home/agent/workspaces/mats/work/run_on_l4.sh "$RID" 0a36abb "$CMD"
