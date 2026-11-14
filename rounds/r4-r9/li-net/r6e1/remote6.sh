#!/bin/bash
# Runs ON the L4 VM as root. /root/soup = frozen copy staged by stage6.sh. Usage: remote6.sh TAG "SEEDS" [extra e1v3.py flags]
# Runs e1v3.py for every seed concurrently; if a seed's run ends without "done" but its theta_real was saved, Part A is retried
# ONCE from that saved theta_real (--load-wm), so a Part A crash does not throw away the 40k-step training.
set -u
TAG=$1; SEEDS=$2; shift 2
cd /root/soup
OUT=/root/runs/r6e1/$TAG; mkdir -p $OUT
t0=$(date +%s)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee $OUT/nvidia-smi.csv
if ! python3 -m pip --version >/dev/null 2>&1; then
  curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python3 /tmp/get-pip.py --break-system-packages -q
fi
python3 -m pip install -q --target /root/pylib torch numpy > /root/pip.log 2>&1 || { tail -20 /root/pip.log; exit 1; }
export PYTHONPATH=/root/pylib:/root/soup SMALLNET_DIR=/root/soup/smallnet R5E1_DIR=/root/soup PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))" | tee $OUT/torch.txt
SETUP=$(( $(date +%s) - t0 ))
for s in $SEEDS; do
  mkdir -p $OUT/s$s
  ( s0=$(date +%s)
    python3 e1v3.py --seed $s --out $OUT/s$s --device cuda --threads 2 "$@" > $OUT/s$s/stdout.txt 2>&1
    rc=$?
    retry=0
    if ! python3 -c "import json,sys; sys.exit(0 if json.load(open('$OUT/s$s/summary.json')).get('done') else 1)" 2>/dev/null \
       && [ -f $OUT/s$s/theta_real_v2.pt ]; then
      retry=1
      python3 e1v3.py --seed $s --out $OUT/s$s --device cuda --threads 2 --load-wm $OUT/s$s/theta_real_v2.pt "$@" \
        > $OUT/s$s/stdout-retry.txt 2>&1
      rc=$?
    fi
    echo "{\"seed\": $s, \"rc\": $rc, \"retry\": $retry, \"sec\": $(( $(date +%s) - s0 ))}" > $OUT/s$s/rc.json ) &
  sleep 2
done
wait
echo "{\"setup_sec\": $SETUP, \"total_sec\": $(( $(date +%s) - t0 )), \"seeds\": \"$SEEDS\", \"args\": \"$*\"}" > $OUT/setup.json
cat $OUT/s*/rc.json
