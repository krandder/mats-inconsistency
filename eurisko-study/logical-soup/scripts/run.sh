#!/usr/bin/env bash
# Run the soup in resumable chunks.
#   scripts/run.sh --seed S --K 2^14 --F 2^12 --epochs N --b LIQ --free-income I \
#     [--held-out-family NAME|0-7 --release EPOCH] [--visible-radius R] \
#     [--fuel F] [--split W] [--w0 W] [--chunk C] --out runs/<name>/
# K = cells, F = questions (powers of two, "2^n" or a number). Writes
# config.json, metrics.csv (header + one row per epoch), soup.dump (state
# after the last finished chunk) and timing.csv (per chunk wall time and
# instructions/s). Rerunning the same command resumes after the last chunk;
# a larger --epochs extends the run. Kill it at any time: the unfinished
# chunk is simply redone. SOUP_CMD overrides the engine (default: bun on
# build/soup.js), e.g. a native-lane binary taking the same arguments.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SEED=1 K=4096 F=4 EPOCHS=10 B=64 INCOME=0 HELD=none RELEASE=none RADIUS=all
FUEL=256 SPLIT=2097152 W0=1048576 CHUNK=50 OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --seed) SEED=$2;; --K) K=$2;; --F) F=$2;; --epochs) EPOCHS=$2;; --b) B=$2;;
    --free-income) INCOME=$2;; --held-out-family) HELD=$2;; --release) RELEASE=$2;;
    --visible-radius) RADIUS=$2;; --fuel) FUEL=$2;; --split) SPLIT=$2;; --w0) W0=$2;;
    --chunk) CHUNK=$2;; --out) OUT=$2;;
    -h|--help) sed -n '2,13p' "$0"; exit 0;;
    *) echo "run.sh: unknown flag $1" >&2; exit 2;;
  esac
  shift 2
done
[ -n "$OUT" ] || { echo "run.sh: --out runs/<name>/ is required" >&2; exit 2; }

log2() {  # "2^n" or a power of two -> n
  local v=$1
  if [[ $v == 2^* ]]; then echo "${v#2^}"; return; fi
  local n=0; while [ $((1 << n)) -lt "$v" ]; do n=$((n + 1)); done
  [ $((1 << n)) -eq "$v" ] || { echo "run.sh: $v is not a power of two" >&2; exit 2; }
  echo $n
}
LOGK=$(log2 "$K"); LOGQ=$(log2 "$F")
[ "$LOGQ" -ge 2 ] || { echo "run.sh: --F must be at least 4 (one block)" >&2; exit 2; }
case "$HELD" in
  none) HF=4294967295;; periodic) HF=0;; lfsr) HF=1;; ca30) HF=2;; ca110) HF=3;;
  prime) HF=4;; pi) HF=5;; and) HF=6;; or) HF=7;; [0-7]) HF=$HELD;;
  *) echo "run.sh: unknown family $HELD" >&2; exit 2;;
esac
[ "$RELEASE" = none ] && REL=4294967295 || REL=$RELEASE
[ "$RADIUS" = all ] && RAD=4294967295 || RAD=$RADIUS

mkdir -p "$OUT"
BUILD="$ROOT/build/soup.js"
if [ -z "${SOUP_CMD:-}" ]; then
  make -C "$ROOT" --no-print-directory build/soup.js >/dev/null
  SOUP_CMD="bun $BUILD"
fi
CONF=$(printf '{"seed": %s, "K": %s, "logk": %s, "F": %s, "logq": %s, "b": %s, "free_income": %s, "held_out_family": "%s", "release": "%s", "visible_radius": "%s", "fuel": %s, "split": %s, "w0": %s, "chunk": %s, "engine": "%s", "commit": "%s"}' \
  "$SEED" $((1 << LOGK)) "$LOGK" $((1 << LOGQ)) "$LOGQ" "$B" "$INCOME" "$HELD" "$RELEASE" "$RADIUS" "$FUEL" "$SPLIT" "$W0" "$CHUNK" "$SOUP_CMD" \
  "$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)")
key() { python3 -c 'import json,sys; d=json.load(sys.stdin); d.pop("commit",None); d.pop("chunk",None); d.pop("engine",None); print(json.dumps(d,sort_keys=True))'; }
if [ -f "$OUT/config.json" ]; then
  [ "$(key < "$OUT/config.json")" = "$(echo "$CONF" | key)" ] || {
    echo "run.sh: $OUT/config.json differs from these flags; use a new --out" >&2; exit 2; }
else
  echo "$CONF" > "$OUT/config.json"
fi

# next epoch stored in the snapshot (bytes 24..27, little-endian)
snap_epoch() { od -An -tu4 -j24 -N4 "$1" | tr -d ' '; }
DONE=0
if [ -f "$OUT/soup.dump" ]; then
  DONE=$(snap_epoch "$OUT/soup.dump")
  # drop rows written after the snapshot (an interrupted append)
  head -n $((DONE + 1)) "$OUT/metrics.csv" > "$OUT/metrics.csv.tmp" && mv "$OUT/metrics.csv.tmp" "$OUT/metrics.csv"
  echo "run.sh: resuming $OUT at epoch $DONE"
else
  rm -f "$OUT/metrics.csv" "$OUT/timing.csv"
fi
[ -f "$OUT/timing.csv" ] || echo "first_epoch,epochs,wall_s,instructions,instr_per_s" > "$OUT/timing.csv"

while [ "$DONE" -lt "$EPOCHS" ]; do
  N=$(( EPOCHS - DONE < CHUNK ? EPOCHS - DONE : CHUNK ))
  IN="-"; [ -f "$OUT/soup.dump" ] && IN="$OUT/soup.dump"
  t0=$(date +%s.%N)
  $SOUP_CMD "$LOGK" "$N" "$SEED" "$FUEL" "$LOGQ" "$SPLIT" "$INCOME" "$W0" "$B" \
    "$HF" "$REL" "$RAD" "$IN" "$OUT/soup.dump.new" > "$OUT/chunk.csv"
  t1=$(date +%s.%N)
  if grep -q '^error' "$OUT/chunk.csv"; then cat "$OUT/chunk.csv" >&2; exit 1; fi
  ROWS=$(grep -c '^[0-9]' "$OUT/chunk.csv" || true)
  [ "$ROWS" -eq "$N" ] || { echo "run.sh: chunk at epoch $DONE wrote $ROWS rows, expected $N" >&2; exit 1; }
  cat "$OUT/chunk.csv" >> "$OUT/metrics.csv"
  mv "$OUT/soup.dump.new" "$OUT/soup.dump"
  rm -f "$OUT/chunk.csv"
  OPS=$(grep '^[0-9]' "$OUT/metrics.csv" | tail -n "$N" | awk -F, '{s += $5} END {printf "%d", s}')
  python3 - "$OUT/timing.csv" "$DONE" "$N" "$t0" "$t1" "$OPS" <<'PY'
import sys
f, done, n, t0, t1, ops = sys.argv[1:]
w = float(t1) - float(t0)
open(f, "a").write(f"{done},{n},{w:.2f},{ops},{int(ops) / w:.0f}\n")
print(f"run.sh: epochs {done}..{int(done) + int(n) - 1} in {w:.1f}s, {int(ops) / w:,.0f} instr/s", flush=True)
PY
  DONE=$((DONE + N))
done
echo "run.sh: $OUT complete at $DONE epochs"
