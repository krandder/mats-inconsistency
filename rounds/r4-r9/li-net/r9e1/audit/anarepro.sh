#!/bin/bash
# R9-E1 audit: re-run the REGISTERED analyze9.py and numbers9.py (hash-checked staged copies) on copies of the three
# summary.json files in a scratch dir and compare their outputs with li-net/r9e1/results/main/analysis/. Read-only for li-net.
# The scratch copies (audit/tmpsrc, audit/anarepro) are moved to the trash afterwards (trash-put, restorable 7 days).
set -eu
A=/home/agent/workspaces/fable-strategy/scratch/r9e1-audit/audit
R=/home/agent/workspaces/eurisko-study/li-net/r9e1/results/main
S=$A/tmpsrc
W=$A/anarepro
mkdir -p "$S" "$W/results/main"
cp /tmp/r9e1/src/analyze9.py /tmp/r9e1/src/numbers9.py "$S/"
( cd "$S" && grep -E ' (analyze9|numbers9)\.py$' /tmp/r9e1/src/MANIFEST.sha256 | sha256sum -c --quiet )
for s in 101 102 103; do
  mkdir -p "$W/results/main/s$s"
  cp "$R/s$s/summary.json" "$W/results/main/s$s/"
done
export PYTHONPATH=/tmp/r4e3a/pylib PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MPLCONFIGDIR=$W/.mpl R9DIR=$W
python3 "$S/analyze9.py" main > /dev/null
python3 "$S/numbers9.py" main > "$W/results/main/analysis/numbers.txt"
for f in verdicts.md numbers.txt anytime_curves.csv d3_weights.csv d3_mse.csv; do
  if cmp -s "$W/results/main/analysis/$f" "$R/analysis/$f"; then echo "$f identical"; else echo "$f DIFFERS"; fi
done
python3 -c "import json; a=json.load(open('$W/results/main/analysis/results.json')); b=json.load(open('$R/analysis/results.json')); print('results.json equal as JSON:', a == b)"
