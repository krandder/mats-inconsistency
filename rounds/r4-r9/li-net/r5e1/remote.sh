#!/bin/bash
# Runs ON the L4 VM as root. /root/soup = frozen copy of li-net/r5e1 (*.py, *.sh) + smallnet/netdef.py (staged by stage.sh).
# Usage: remote.sh TAG "SEEDS" [extra run.py flags...]
#   runs run.py for every seed in SEEDS concurrently on the one GPU; outputs /root/runs/r5e1/TAG/s<seed>/
set -u
TAG=$1; SEEDS=$2; shift 2
cd /root/soup
SCRIPT=run.py; [[ $TAG == v2* ]] && SCRIPT=partA2.py      # amendment (PREREG-AMENDMENTS.md): Part A v2, bounded-context reads
OUT=/root/runs/r5e1/$TAG; mkdir -p $OUT
t0=$(date +%s)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee $OUT/nvidia-smi.csv
if ! python3 -m pip --version >/dev/null 2>&1; then
  curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python3 /tmp/get-pip.py --break-system-packages -q
fi
python3 -m pip install -q --target /root/pylib torch numpy > /root/pip.log 2>&1 || { tail -20 /root/pip.log; exit 1; }
export PYTHONPATH=/root/pylib:/root/soup SMALLNET_DIR=/root/soup/smallnet PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))" | tee $OUT/torch.txt
SETUP=$(( $(date +%s) - t0 ))
echo "{\"setup_sec\": $SETUP, \"seeds\": \"$SEEDS\", \"args\": \"$*\"}" > $OUT/setup.json
for s in $SEEDS; do
  mkdir -p $OUT/s$s
  ( s0=$(date +%s)
    python3 $SCRIPT --seed $s --out $OUT/s$s --device cuda --threads 2 "$@" > $OUT/s$s/stdout.txt 2>&1
    rc=$?
    echo "{\"seed\": $s, \"rc\": $rc, \"sec\": $(( $(date +%s) - s0 ))}" > $OUT/s$s/rc.json ) &
  sleep 2
done
wait
echo "{\"setup_sec\": $SETUP, \"total_sec\": $(( $(date +%s) - t0 )), \"seeds\": \"$SEEDS\", \"args\": \"$*\"}" > $OUT/setup.json
cat $OUT/s*/rc.json
