#!/bin/bash
# Runs ON the GPU VM as root. /root/soup = frozen copy: harness/env (unmodified), smallnet/*.py (round 1, unmodified),
# wm2/*.py, data-t1/, data-t2/, jobs.txt. Usage: remote.sh JOBFILE [PAR] [MINUTES]. One job.py argument list per line;
# PAR jobs at a time on the one GPU, by remote_sched.py (a forecaster starts as soon as its WM checkpoint exists). WM checkpoints go to /root/ckpt (never synced). Status per finished job:
# /root/runs/wm2/status/<host>.jsonl.
set -u
cd /root/soup
JOBS=$1; PAR=${2:-6}; MIN=${3:-0}
H=$(hostname)
mkdir -p /root/runs/wm2/status /root/logs /root/ckpt
ST=/root/runs/wm2/status/$H.jsonl
t0=$(date +%s)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if ! python3 -m pip --version >/dev/null 2>&1; then
  curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python3 /tmp/get-pip.py --break-system-packages -q
fi
python3 -m pip install -q --target /root/pylib torch numpy > /root/pip.log 2>&1 || { tail -20 /root/pip.log; exit 1; }
export PYTHONPATH=/root/pylib LIR_HARNESS=/root/soup/harness WM2_DATA=/root/soup WM2_SMALLNET=/root/soup/smallnet \
  PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
(( MIN > 0 )) && export SMALLNET_DEADLINE=$(( t0 + MIN * 60 - 120 ))
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "{\"host\": \"$H\", \"event\": \"setup\", \"sec\": $(( $(date +%s) - t0 )), \"torch\": \"$(python3 -c 'import torch; print(torch.__version__)')\"}" >> $ST
python3 /root/soup/remote_sched.py "$JOBS" "$PAR" "$ST"
