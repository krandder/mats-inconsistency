#!/usr/bin/env bash
# Freeze all executable code before launching three concurrent supervised runs.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
if [[ ${1:-} == --dry-run ]]; then
  source infra/lib.sh
  group=$(spend_usd pool-sddpv); total=$(spend_usd)
  printf 'Three new L4 reservations: $15.66; group existing=$%s cap=$40; project existing=$%s cap=$200\n' "$group" "$total"
  awk -v g="$group" -v t="$total" 'BEGIN{exit (g+15.66>40 || t+15.66>200)}'
  for seed in 1 2 3; do scripts/gpu-phase3.sh --dry-run "$seed"; done
  exit 0
fi
if [[ ${1:-} == --frozen-run ]]; then
  frozen=$PWD
  ROOT=${2:?control root}
  [[ ${GPU_PHASE3_FROZEN:-0} == 1 && -f source-commit.txt ]] || exit 2
else
  [[ -z $(git diff HEAD -- src scripts infra Makefile tests) ]] || { echo 'commit executable changes before launch' >&2; exit 2; }
  sha=$(git rev-parse HEAD)
  frozen="$ROOT/build/phase3-runtime-${sha:0:12}"
  [[ ! -e $frozen ]] || { echo "immutable launch directory already exists: $frozen" >&2; exit 2; }
  mkdir -p "$frozen"
  git archive "$sha" | tar -x -C "$frozen"
  printf '%s\n' "$sha" > "$frozen/source-commit.txt"
  chmod -R a-w "$frozen/infra" "$frozen/scripts" "$frozen/src"
  exec env GPU_CONTROL_ROOT="$ROOT" GPU_PHASE3_FROZEN=1 "$frozen/scripts/launch-gpu-phase3.sh" --frozen-run "$ROOT"
fi
export GPU_CONTROL_ROOT="$ROOT" GPU_PHASE3_FROZEN=1
mkdir -p "$ROOT/infra/state/phase3-logs"
pids=()
for seed in 1 2 3; do
  "$frozen/scripts/gpu-phase3.sh" "$seed" >> "$ROOT/infra/state/phase3-logs/seed$seed.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
python3 "$frozen/scripts/gpu-phase3-finish.py" "$ROOT" || exit 1
exit "$rc"
