#!/bin/bash
# M1(b) GPU job: lr pilot for plain SGD, then adam/sgd/sgdwd x seeds 1-3 in parallel.
set -u
PY=${PY:-/opt/tn/bin/python}
OUT=${OUT:-/root/out}
mkdir -p "$OUT/pilot" "$OUT/runs"
nvidia-smi -L > "$OUT/gpu.txt" 2>&1 || true
STEPS=${STEPS:-10000}; MAXMIN=${MAXMIN:-62}
# pilot: largest eta in {0.3,1,3} with a finite loss for 300 steps (PREREG-w2)
for lr in 0.3 1 3; do
  $PY train_opt.py --opt sgd --lr $lr --steps 300 --snap 100 --seed 1 --out "$OUT/pilot/lr$lr" > "$OUT/pilot/lr$lr.log" 2>&1 &
done
wait
ETA=0.3
for lr in 0.3 1 3; do
  if grep -q "done in" "$OUT/pilot/lr$lr.log" && ! grep -q "NaN" "$OUT/pilot/lr$lr.log"; then ETA=$lr; fi
done
WD=$($PY -c "print(1.0/($ETA*$STEPS))")
echo "eta=$ETA wd=$WD steps=$STEPS" | tee "$OUT/choice.txt"
for s in 1 2 3; do
  $PY train_opt.py --opt adam --lr 1e-3 --steps $STEPS --seed $s --max-min $MAXMIN --out "$OUT/runs" > "$OUT/runs/adam-s$s.log" 2>&1 &
  $PY train_opt.py --opt sgd --lr $ETA --steps $STEPS --seed $s --max-min $MAXMIN --out "$OUT/runs" > "$OUT/runs/sgd-s$s.log" 2>&1 &
  $PY train_opt.py --opt sgdwd --lr $ETA --wd $WD --steps $STEPS --seed $s --max-min $MAXMIN --out "$OUT/runs" > "$OUT/runs/sgdwd-s$s.log" 2>&1 &
done
wait
$PY analyze_w2.py "$OUT/runs" "$OUT/analysis" > "$OUT/analysis.txt" 2>&1 || true
echo done > "$OUT/DONE"
