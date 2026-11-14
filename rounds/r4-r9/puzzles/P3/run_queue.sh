#!/bin/bash
# Runs t1_queue.txt ("NAME args...") with P parallel single-thread jobs; results -> $OUT/NAME.json, logs -> $OUT/NAME.log.
# Usage: run_queue.sh QUEUE OUT P PYTHON [extra t1.py args, e.g. a smoke override]
Q=$1; OUT=$2; P=$3; PY=$4; shift 4
DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"; touch "$OUT/JOB_RUNNING"
export OUT PY DIR EXTRA="$*"
run_one() {
  local name=$1; shift
  [[ -f $OUT/$name.json ]] && return 0
  OMP_NUM_THREADS=1 "$PY" "$DIR/t1.py" "$@" --threads 1 --out "$OUT/$name.json.tmp" $EXTRA >"$OUT/$name.log" 2>&1 \
    && mv "$OUT/$name.json.tmp" "$OUT/$name.json"
}
export -f run_one
xargs -P "$P" -L 1 bash -c 'run_one "$@"' _ < "$Q"
rm -f "$OUT/JOB_RUNNING"; date -u +%FT%TZ > "$OUT/JOB_DONE"
