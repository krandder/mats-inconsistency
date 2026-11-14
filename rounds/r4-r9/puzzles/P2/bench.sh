#!/bin/bash
# Throughput check on a P2 VM before the main queue (run as root in /root/job):
#   bash bench.sh nomps|mps  -> /root/bench/<mode>/summary.txt
# 4 concurrent 101-step training runs (M-tree, U-min with Viterbi labels from step 0, U, M-tile),
# then one n=33 eval of the M-tree run while nothing else runs.
mode=$1
B=/root/bench/$mode
mkdir -p "$B"
PY=/opt/venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/root/job
if [[ $mode == mps ]]; then nvidia-cuda-mps-control -d >>"$B/mps.log" 2>&1; else echo quit | nvidia-cuda-mps-control >>"$B/mps.log" 2>&1; fi
t0=$(date +%s)
for arm in M-tree U-min U M-tile; do
  extra=""; [[ $arm == U-min ]] && extra="--warm 0"
  $PY train.py --arm $arm --seed 0 --steps 101 --ckpt-every 1000 --out "$B/runs" $extra >>"$B/train_$arm.log" 2>&1 &
done
wait
t1=$(date +%s)
$PY eval.py run --ckpt "$B/runs/M-tree_s0_x101.pt" --arm M-tree --seed 0 --cells n33 --out "$B/eval" >>"$B/eval.log" 2>&1
t2=$(date +%s)
{
  echo "mode=$mode train_wall=$((t1 - t0))s eval_n33_wall=$((t2 - t1))s"
  for arm in M-tree U-min U M-tile; do
    python3 - "$B/train_$arm.log" "$arm" <<'PY'
import json, sys
h = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
bank = [x for x in h if "bank_sec" in x]
st = {x["step"]: x["sec"] for x in h if "step" in x}
print(sys.argv[2], "bank_sec", bank[0]["bank_sec"] if bank else "?", "s/step(50..100)",
      round((st[100] - st[50]) / 50, 4) if 100 in st and 50 in st else "?")
PY
  done
  grep -E "^test\||^dev\||EVAL_DONE" "$B/eval.log" | awk '{print $1, $NF}' | tr '\n' ' '; echo
} >"$B/summary.txt"
cat "$B/summary.txt"
