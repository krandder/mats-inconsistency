#!/usr/bin/env bash
# Usage: launch.sh <run-id> <script-on-vm> [extra env assignments]
# Packs the code, runs <script-on-vm> on one L4 (SPOT by default; GPU_PROVISIONING=STANDARD for on-demand),
# syncs /root/runs/<run-id> back to /tmp/l4runs/<run-id>/runs/<run-id>/ and deletes the VM (run_on_l4.sh).
set -euo pipefail
RID=$1; SCRIPT=$2; shift 2
cd /home/agent/workspaces/mats/work/estimability
FILES=$(ls world_small.py model.py prop.py mcstats.py train.py evaluate.py validate_init.py test_prop.py oputil.py gpu_pilot.sh gpu_sweep.sh summarize.py PREREG.md code_sha256.txt 2>/dev/null || true)
B=$(tar cz $FILES | base64 -w0)
CMD="mkdir -p /root/runs/$RID && cd /root/runs/$RID && echo $B | base64 -d | tar xz && nvidia-smi -L && /root/venv/bin/pip install -q scipy opt_einsum && export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore OUT=/root/runs/$RID/out $* && bash $SCRIPT > /root/runs/$RID/job.log 2>&1; tail -5 /root/runs/$RID/job.log"
GPU_PROVISIONING=${GPU_PROVISIONING:-SPOT} GPU_RUN_TIMEOUT=${GPU_RUN_TIMEOUT:-3600s} /home/agent/workspaces/mats/work/run_on_l4.sh "$RID" 0a36abb "$CMD"
