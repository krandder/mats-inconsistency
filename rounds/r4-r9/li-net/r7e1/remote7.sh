#!/bin/bash
# Runs ON the L4 VM as root. /root/soup = frozen copy staged by stage7.sh. Usage: remote7.sh TAG "SEEDS" [extra e1v4.py flags]
# Runs e1v4.py for every seed concurrently. Each seed saves theta_real to /root/runs/r7e1/ckpt/s<seed>/theta_real_v2.npz
# (driver7.sh syncs it off the VM). If /root/soup/ckpt/s<seed>/theta_real_v2.npz was staged (recovery), that is loaded
# instead of training. If a seed's run ends without "done" but its theta_real was saved, it is retried ONCE from it.
set -u
TAG=$1; SEEDS=$2; shift 2
cd /root/soup
OUT=/root/runs/r7e1/$TAG; CK=/root/runs/r7e1/ckpt; mkdir -p $OUT $CK
t0=$(date +%s)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee $OUT/nvidia-smi.csv
if ! python3 -m pip --version >/dev/null 2>&1; then
  curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python3 /tmp/get-pip.py --break-system-packages -q
fi
python3 -m pip install -q --target /root/pylib torch numpy > /root/pip.log 2>&1 || { tail -20 /root/pip.log; exit 1; }
export PYTHONPATH=/root/pylib:/root/soup SMALLNET_DIR=/root/soup/smallnet R5E1_DIR=/root/soup R6E1_DIR=/root/soup
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))" | tee $OUT/torch.txt
SETUP=$(( $(date +%s) - t0 ))
for s in $SEEDS; do
  mkdir -p $OUT/s$s
  ( s0=$(date +%s)
    PRE=/root/soup/ckpt/s$s/theta_real_v2.npz
    pre=false
    LOAD=()
    if [ -f $PRE ]; then pre=true; LOAD=(--load-wm $PRE); fi
    python3 e1v4.py --seed $s --out $OUT/s$s --ckpt-dir $CK --device cuda --threads 2 ${LOAD[@]+"${LOAD[@]}"} "$@" \
      > $OUT/s$s/stdout.txt 2>&1
    rc=$?
    retry=0
    if ! python3 -c "import json,sys; sys.exit(0 if json.load(open('$OUT/s$s/summary.json')).get('done') else 1)" 2>/dev/null; then
      W=$PRE; [ -f $CK/s$s/theta_real_v2.npz ] && W=$CK/s$s/theta_real_v2.npz
      if [ -f $W ]; then
        retry=1
        python3 e1v4.py --seed $s --out $OUT/s$s --ckpt-dir $CK --device cuda --threads 2 --load-wm $W "$@" \
          > $OUT/s$s/stdout-retry.txt 2>&1
        rc=$?
      fi
    fi
    echo "{\"seed\": $s, \"rc\": $rc, \"retry\": $retry, \"preloaded\": $pre, \"sec\": $(( $(date +%s) - s0 ))}" > $OUT/s$s/rc.json ) &
  sleep 2
done
wait
echo "{\"setup_sec\": $SETUP, \"total_sec\": $(( $(date +%s) - t0 )), \"seeds\": \"$SEEDS\", \"args\": \"$*\"}" > $OUT/setup.json
cat $OUT/s*/rc.json
