#!/bin/bash
# Runs on the GPU VM (cwd = synced code dir). Installs torch if missing, trains 3 arms x seeds in parallel.
# Usage: [ARMS="A B C"] [OUT=/root/runs/blocklocal] run_bl.sh STEPS MAXMIN [SEEDS...]
set -u
STEPS=${1:-30000}; MAXMIN=${2:-45}; shift 2 || true
SEEDS=${@:-1 2 3}
OUT=${OUT:-/root/runs/blocklocal}; mkdir -p "$OUT"; ARMS=${ARMS:-A B C}
export PYTHONPATH=/root/pylib
[ -d ref ] && cp ref/*-log.json "$OUT/"   # C logs: skip/birth schedule for arm R
if ! python3 -c "import torch" 2>/dev/null; then
  curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python3 /tmp/get-pip.py -q --break-system-packages >"$OUT/pip.log" 2>&1
  python3 -m pip install -q --break-system-packages --target /root/pylib torch numpy scipy --index-url https://download.pytorch.org/whl/cu129 --extra-index-url https://pypi.org/simple >>"$OUT/pip.log" 2>&1
fi
python3 -c "import torch;print(torch.__version__, torch.cuda.is_available())" > "$OUT/env.txt" 2>&1
nvidia-smi -L >> "$OUT/env.txt"
for a in $ARMS; do for s in $SEEDS; do
  DEVICE=cuda THREADS=1 MAXMIN=$MAXMIN python3 blocklocal.py $a $s $STEPS "$OUT" > "$OUT/$a-s$s.log" 2>&1 &
done; done
wait
echo done > "$OUT/DONE"
