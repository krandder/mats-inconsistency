#!/bin/bash
# E-CA finalisation: copy raw per-run files (json/csv) under results/raw, run the analyses (primary, replication, T=8),
# rebuild the GPU ledger, assemble RESULTS.md and render eca.html. Idempotent.
set -eu
E=/home/agent/workspaces/eurisko-study/li-net/eca
cd $E
mkdir -p results/raw results/replication results/t8
for b in main2 main t8; do
  [ -d /tmp/eca/runs/$b ] || continue
  rsync -a --include='*/' --include='*.json' --include='*.csv' --exclude='*' /tmp/eca/runs/$b/ results/raw/$b/
done
cp /tmp/eca/ecarun-as-run-0135Z.py results/raw/ecarun-as-run-0135Z.py
export PYTHONPATH=/tmp/smallnet/pylib ECA_QUIET=1
ECA_OUT=$E/results python3 analyze.py $E/results/raw/main2 $E/results/raw/main > /tmp/eca/logs/analyze-final.log 2>&1
ECA_OUT=$E/results/replication python3 analyze.py $E/results/raw/main > /tmp/eca/logs/analyze-replication.log 2>&1
ECA_OUT=$E/results/t8 python3 analyze.py $E/results/raw/t8 > /tmp/eca/logs/analyze-t8.log 2>&1
python3 gpu/ledger.py
python3 build_results.py
./render.sh
du -sh results
