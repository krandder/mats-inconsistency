#!/usr/bin/env bash
# Copy E21's synced outputs (/tmp/l4runs/<rid>/runs/<rid>/w{W}_s{S}/) into runs/e21_w{W}_s{S}/ (copies; nothing is deleted).
set -euo pipefail
RID=${1:-cmb-e21-w3}
SRC=/tmp/l4runs/$RID/runs/$RID
cd /home/agent/workspaces/mats/work/cm-interp-B
for d in "$SRC"/w*_s*; do
  [ -d "$d" ] || continue
  n=$(basename "$d")
  if [ -f "$d/meta.json" ] && [ ! -e "runs/e21_$n" ]; then
    cp -r "$d" "runs/e21_$n"; cp "$SRC/$n.log" "runs/e21_$n.log" 2>/dev/null || true
    echo "copied $n"
  elif [ ! -f "$d/meta.json" ]; then
    echo "skip $n (no meta.json: training did not finish)"
  fi
done
