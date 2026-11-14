#!/usr/bin/env bash
# Run a matrix of scripts/run.sh invocations described by a JSON file.
#   scripts/matrix.sh [matrices/phase3.json] [-j N] [--dry-run] [--engine CMD]
#                     [--name NAME] [--only RE] [--epochs N]
# Each run goes to runs/<name>/k<logK>-f<logF>-s<seed>[-<variant>]/ (name from
# the JSON unless --name). At most N runs at once (default 8; each JS-lane run
# is single-threaded). Resumable: a run whose metrics.csv already has
# epochs+1 lines is skipped; an unfinished one resumes from its snapshot
# (run.sh). Runs start largest K first. --dry-run prints the commands and
# what would be skipped.
# --engine sets SOUP_CMD for every run (e.g. the native binary on the GPU VM,
# via infra/gpu-run.sh -- scripts/matrix.sh ... --engine build/soup-native).
# Without --engine the JS engine is frozen to runs/<name>/soup.js on first start.
# --only runs just the cells whose directory name matches the regex RE.
# Per-run stdout/stderr: <run>/run.log. Matrix-level log: runs/<name>/matrix.log.
# JSON keys: name, epochs, chunk, base {run.sh flag: value}, K, F, seeds,
# variants [{name, K, F, seeds?, args {flag: value}, bootstrap {free-income, epochs}}].
# A bootstrap variant first runs <run>/bootstrap/ with that free income for
# that many epochs, then continues in <run>/ with the base flags from its
# snapshot (run.sh resumes it); <run>/bootstrap.json records the handoff.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SPEC="$ROOT/matrices/phase3.json" JOBS=8 DRY=0 NAME="" ONLY="" EPOCHS=""
while [ $# -gt 0 ]; do
  case "$1" in
    -j|--jobs) JOBS=$2; shift;;
    --dry-run) DRY=1;;
    --engine) export SOUP_CMD=$2; shift;;
    --name) NAME=$2; shift;;
    --only) ONLY=$2; shift;;
    --epochs) EPOCHS=$2; shift;;
    -h|--help) sed -n '2,20p' "$0"; exit 0;;
    -*) echo "matrix.sh: unknown flag $1" >&2; exit 2;;
    *) SPEC=$1;;
  esac
  shift
done
[ -f "$SPEC" ] || { echo "matrix.sh: no matrix file $SPEC" >&2; exit 2; }

# Expand the spec: one line per run, tab-separated fields:
# dir, epochs, bootstrap income, bootstrap epochs, run.sh flags (space-separated,
# values must not contain spaces)
PLAN=$(python3 - "$SPEC" "$NAME" "$EPOCHS" "$ONLY" <<'PY'
import json, re, sys
spec, name, epochs, only = sys.argv[1:]
m = json.load(open(spec))
name = name or m["name"]
epochs = int(epochs or m["epochs"])
base = dict(m.get("base", {}))
def lg(v):
    v = str(v)
    if v.startswith("2^"): return int(v[2:])
    n = int(v); k = n.bit_length() - 1
    if n != 1 << k: sys.exit(f"matrix.sh: {v} is not a power of two")
    return k
def cell(K, F, seed, var=None):
    d = f"k{lg(K)}-f{lg(F)}-s{seed}" + (f"-{var['name']}" if var else "")
    a = dict(base); a.update((var or {}).get("args", {}))
    a.update({"seed": seed, "K": K, "F": F, "epochs": epochs, "chunk": m.get("chunk", 100)})
    for k, v in a.items():
        if not re.fullmatch(r"[A-Za-z0-9^._-]+", f"{k}{v}"):
            sys.exit(f"matrix.sh: bad flag/value {k}={v}")
    boot = (var or {}).get("bootstrap") or {}
    flags = " ".join(f"--{k} {v}" for k, v in a.items())
    return d, boot.get("free-income", "-"), boot.get("epochs", "-"), flags
cells = [cell(K, F, s) for K in m["K"] for F in m["F"] for s in m["seeds"]]
for v in m.get("variants", []):
    cells += [cell(v["K"], v["F"], s, v) for s in v.get("seeds", m["seeds"])]
# largest K first: the long runs start at once, the short ones fill the gaps
cells.sort(key=lambda c: -int(c[0].split("-")[0][1:]))
seen = set()
for d, bi, be, flags in cells:
    if d in seen: sys.exit(f"matrix.sh: duplicate run {d}")
    seen.add(d)
    if only and not re.search(only, d): continue
    print(f"runs/{name}/{d}\t{epochs}\t{bi}\t{be}\t{flags}")
PY
)
[ -n "$PLAN" ] || { echo "matrix.sh: no runs selected" >&2; exit 2; }
MNAME=$(echo "$PLAN" | head -1 | cut -d/ -f2)

rows() { if [ -f "$1/metrics.csv" ]; then grep -c . "$1/metrics.csv" || true; else echo 0; fi; }

# one run: $1 dir, $2 epochs, $3 bootstrap income, $4 bootstrap epochs, $5 flags
run_one() {
  local dir=$1 epochs=$2 bi=$3 be=$4 t0 t1 rc=0
  local -a flags bflags
  read -ra flags <<< "$5"
  mkdir -p "$ROOT/$dir"
  t0=$(date +%s)
  {
    if [ "$bi" != - ] && [ ! -f "$ROOT/$dir/soup.dump" ]; then
      bflags=("${flags[@]}")
      for i in "${!bflags[@]}"; do
        case "${bflags[$i]}" in
          --free-income) bflags[$((i + 1))]=$bi;;
          --epochs) bflags[$((i + 1))]=$be;;
        esac
      done
      "$ROOT/scripts/run.sh" "${bflags[@]}" --out "$ROOT/$dir/bootstrap" &&
      cp "$ROOT/$dir/bootstrap/soup.dump" "$ROOT/$dir/bootstrap/metrics.csv" "$ROOT/$dir/bootstrap/timing.csv" "$ROOT/$dir/" &&
      printf '{"free_income": %s, "epochs": %s, "from": "bootstrap/"}\n' "$bi" "$be" > "$ROOT/$dir/bootstrap.json"
    fi &&
    "$ROOT/scripts/run.sh" "${flags[@]}" --out "$ROOT/$dir"
  } >> "$ROOT/$dir/run.log" 2>&1 || rc=$?
  t1=$(date +%s)
  echo "$(date -u +%FT%TZ) $dir rc=$rc wall_s=$((t1 - t0)) rows=$(rows "$ROOT/$dir")" | tee -a "$ROOT/runs/$MNAME/matrix.log"
  return $rc
}

TODO=() SKIP=0
while IFS=$'\t' read -r dir epochs bi be flags; do
  if [ "$(rows "$ROOT/$dir")" -ge $((epochs + 1)) ]; then
    SKIP=$((SKIP + 1)); [ $DRY = 1 ] && echo "# done: $dir"; continue
  fi
  TODO+=("$dir"$'\t'"$epochs"$'\t'"$bi"$'\t'"$be"$'\t'"$flags")
  if [ $DRY = 1 ]; then
    [ "$bi" != - ] && echo "# first $be epochs at free income $bi into $dir/bootstrap, then:"
    echo "${SOUP_CMD:+SOUP_CMD=$(printf %q "$SOUP_CMD") }scripts/run.sh $flags --out $dir"
  fi
done <<< "$PLAN"
echo "matrix.sh: $MNAME: ${#TODO[@]} to run, $SKIP already done, -j $JOBS${SOUP_CMD:+, engine $SOUP_CMD}" >&2
[ $DRY = 1 ] && exit 0
[ ${#TODO[@]} -gt 0 ] || exit 0
mkdir -p "$ROOT/runs/$MNAME"
cp "$SPEC" "$ROOT/runs/$MNAME/matrix.json"
# Without --engine, freeze the JS engine once per matrix (runs/<name>/soup.js)
# so a rebuild of build/soup.js mid-matrix cannot mix engines; resume reuses it.
if [ -z "${SOUP_CMD:-}" ]; then
  if [ ! -f "$ROOT/runs/$MNAME/soup.js" ]; then
    make -C "$ROOT" --no-print-directory build/soup.js >/dev/null
    cp "$ROOT/build/soup.js" "$ROOT/runs/$MNAME/soup.js"
    { git -C "$ROOT" rev-parse --short HEAD; git -C "$ROOT" status --short src; } > "$ROOT/runs/$MNAME/soup.js.commit"
  fi
  # K >= 2^15 overflows the default JS stack ("bend: memory fault"); a 4 GB
  # stack rlimit plus JSC's per-thread cap (below 2^32) lets K = 2^16 run.
  export SOUP_CMD="env BUN_JSC_maxPerThreadStackUsage=4000000000 bun $ROOT/runs/$MNAME/soup.js"
  ulimit -s 4194304 2>/dev/null || ulimit -s unlimited 2>/dev/null || true
fi
cd "$ROOT"
FAIL=0
for job in "${TODO[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n || FAIL=$((FAIL + 1)); done
  IFS=$'\t' read -r dir epochs bi be flags <<< "$job"
  run_one "$dir" "$epochs" "$bi" "$be" "$flags" &
done
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do wait -n || FAIL=$((FAIL + 1)); done
echo "matrix.sh: $MNAME finished, $FAIL failed (see runs/$MNAME/matrix.log)" >&2
[ $FAIL = 0 ]
