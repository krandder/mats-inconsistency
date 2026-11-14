#!/usr/bin/env bash
# Copy a finished VM's outputs (/tmp/l4runs/<rid>/runs/<rid>/) into runs/eNN/ by name prefix (e37-e49).
# Existing run directories are not overwritten. usage: gpu/collect_b4.sh <rid>
set -euo pipefail
RID=$1
SRC=/tmp/l4runs/$RID/runs/$RID
cd /home/agent/workspaces/mats/work/cm-interp-B
[ -d "$SRC" ] || { echo "no outputs at $SRC"; exit 1; }
for d in "$SRC"/e3[789]_* "$SRC"/e4[0-9]_*; do
  [ -d "$d" ] || continue
  n=$(basename "$d"); e=${n%%_*}; mkdir -p runs/$e
  if [ -e runs/$e/$n ]; then echo "exists: runs/$e/$n"; continue; fi
  cp -r "$d" runs/$e/$n
  [ -f "$SRC/$n.log" ] && cp "$SRC/$n.log" runs/$e/$n.log
done
cp "$SRC/jobs.txt" runs/jobs_$RID.txt 2>/dev/null || true
for e in e37 e38 e39 e40 e43; do [ -d runs/$e ] && echo "$e: $(ls -d runs/$e/*/ 2>/dev/null | wc -l) runs, $(ls runs/$e/*/meta.json 2>/dev/null | wc -l) finished"; done
