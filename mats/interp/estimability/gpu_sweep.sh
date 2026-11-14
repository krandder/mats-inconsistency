#!/usr/bin/env bash
# Job on one L4 for the depths in $DEPTHS: (1) step timing and the preregistered pen_every rule, (2) op tests (CPU,
# background), (3) optional pre-training validation, (4) training runs, (5) evaluation of every trained model,
# random-features model and untrained init, (6) summary. Env: OUT, DEPTHS ("8 4"), STEPS (3000), P, WALL_H, SKIP_VALIDATE.
set -uo pipefail
STEPS=${STEPS:-3000}; P=${P:-2}; WALL_H=${WALL_H:-4.0}; DEPTHS=${DEPTHS:-"8 4"}; SEEDS=${SEEDS:-"0 1 2"}; ONLY_PEN=${ONLY_PEN:-0}
mkdir -p "$OUT/runs" "$OUT/timing"
sha256sum world_small.py model.py prop.py mcstats.py train.py evaluate.py validate_init.py test_prop.py oputil.py gpu_sweep.sh summarize.py > "$OUT/code_sha256_vm.txt"
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
# ---- (1) timing from init (not results): per-step time = difference between the records at steps 3 and 6
for D in $([ -n "${PEN_EVERY_FIXED:-}" ] || echo $DEPTHS); do
  for K in 4 2; do python train.py --depth $D --lam 3 --K $K --seed 9 --steps 6 --eval_every 3 --out $OUT/timing/d${D}_K$K > $OUT/timing/d${D}_K$K.log 2>&1; done
  python train.py --depth $D --lam 0 --seed 9 --steps 200 --eval_every 100 --out $OUT/timing/d${D}_l0 > $OUT/timing/d${D}_l0.log 2>&1
done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
PEN_EVERY=${PEN_EVERY_FIXED:-}
[ -n "$PEN_EVERY" ] || PEN_EVERY=$(python - "$OUT" "$STEPS" "$P" "$WALL_H" "$DEPTHS" <<'EOF'
import json, sys
out, steps, P, wall, depths = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5].split()
def tstep(c, a, b):
    try:
        log = json.load(open(f'{out}/timing/{c}/meta.json'))['log']
        return (log[-1]['sec'] - log[0]['sec']) / (b - a)
    except Exception:
        return None
serial = {}
for k in (1, 2, 4, 8):
    tot = 0.0; ok = True
    for D in depths:
        t0 = tstep(f'd{D}_l0', 100, 200); t4 = tstep(f'd{D}_K4', 3, 6); t2 = tstep(f'd{D}_K2', 3, 6)
        print(f'depth {D}: s/step lambda0 {t0} K4 {t4} K2 {t2}', file=sys.stderr)
        if None in (t0, t4, t2):
            ok = False; break
        per = lambda t: steps * (t0 + (t - t0) / k)
        tot += 3 * (3 * per(t4) + per(t2) + 2 * steps * t0)
    if not ok:
        print(8); break
    hours = tot / 3600 / min(P, 1.5) + 0.8
    print(f'k={k}: projected {hours:.2f} h', file=sys.stderr)
    if hours <= wall:
        print(k); break
else:
    print(8)
EOF
)
echo "PEN_EVERY=$PEN_EVERY" | tee "$OUT/pen_every.txt"
# ---- (2) op tests, (3) validation at init
(python test_prop.py > "$OUT/test_prop.txt" 2>&1 &)
if [ "${SKIP_VALIDATE:-0}" != 1 ]; then
  python validate_init.py --out "$OUT/validate_init" --n 1048576 --seeds 0 1 2 --stress 1 2 --depths 1 2 4 8 --device cuda \
    > "$OUT/validate_init.log" 2>&1
  echo "validate_init done $(date -u +%H:%M)"
fi
# ---- (4) training
: > arms.txt
for D in $DEPTHS; do for S in $SEEDS; do
  echo "$D 30 4 $S 0" >> arms.txt; echo "$D 3 4 $S 0" >> arms.txt; echo "$D 0.3 4 $S 0" >> arms.txt
  echo "$D 3 2 $S 0" >> arms.txt
done; done
[ "$ONLY_PEN" = 1 ] || for D in $DEPTHS; do for S in $SEEDS; do echo "$D 0 4 $S 0" >> arms.txt; echo "$D 0 4 $S 1" >> arms.txt; done; done
export OUT STEPS PEN_EVERY
cat arms.txt | xargs -P "$P" -L 1 bash -c 'R=$OUT/runs/d$0_l$1_K$2_s$3_f$4; [ -f $R/model.pt ] && exit 0; python train.py --depth $0 --lam $1 --K $2 --seed $3 --freeze_body $4 --steps $STEPS --pen_every $PEN_EVERY --out $R > $R.log 2>&1; echo "TRAINED $R $(date -u +%H:%M) $(tail -1 $R.log | cut -c1-160)"'
# retry sequentially any run that failed (e.g. GPU memory under concurrency)
while read -r D L K S F; do R=$OUT/runs/d${D}_l${L}_K${K}_s${S}_f${F}; [ -f $R/model.pt ] || { python train.py --depth $D --lam $L --K $K --seed $S --freeze_body $F --steps $STEPS --pen_every $PEN_EVERY --out $R > $R.log 2>&1; echo "RETRIED $R $(date -u +%H:%M)"; }; done < arms.txt
# ---- (5) evaluation
ls -d "$OUT"/runs/*/ | xargs -P "$P" -I{} bash -c 'R={}; R=${R%/}; [ -f $R/eval.json ] || python evaluate.py --ckpt $R/model.pt --out $R/eval.json > $R.eval.log 2>&1; case $R in *_l0_K4_*_f0) [ -f $R/eval_init.json ] || python evaluate.py --ckpt $R/model.pt --init --out $R/eval_init.json > $R.evalinit.log 2>&1 ;; esac; echo "EVAL $R $(date -u +%H:%M)"'
python summarize.py "$OUT" > "$OUT/summary.md" 2>&1
echo "SWEEP DONE $(date -u +%H:%M)"
