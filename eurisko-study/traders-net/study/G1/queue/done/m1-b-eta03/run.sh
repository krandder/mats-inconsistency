#!/bin/bash
# M1(b) amendment 1 (exploratory): sgd/sgdwd at eta=0.3 x seeds 1-3.
set -u
PY=${PY:-/opt/tn/bin/python}
OUT=${OUT:-/root/out}
mkdir -p "$OUT/pilot" "$OUT/runs"
nvidia-smi -L > "$OUT/gpu.txt" 2>&1 || true
STEPS=${STEPS:-10000}; MAXMIN=${MAXMIN:-62}; ETA=0.3
WD=$($PY -c "print(1.0/($ETA*$STEPS))")
echo "eta=$ETA wd=$WD steps=$STEPS" | tee "$OUT/choice.txt"
for s in 1 2 3; do
  $PY train_opt.py --opt sgd --lr $ETA --steps $STEPS --seed $s --max-min $MAXMIN --out "$OUT/runs" > "$OUT/runs/sgd-s$s.log" 2>&1 &
  $PY train_opt.py --opt sgdwd --lr $ETA --wd $WD --steps $STEPS --seed $s --max-min $MAXMIN --out "$OUT/runs" > "$OUT/runs/sgdwd-s$s.log" 2>&1 &
done
wait
$PY analyze_w2.py "$OUT/runs" "$OUT/analysis" > "$OUT/analysis.txt" 2>&1 || true
echo done > "$OUT/DONE"
