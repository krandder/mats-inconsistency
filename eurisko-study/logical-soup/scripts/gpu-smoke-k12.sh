#!/usr/bin/env bash
# One bounded VM command for pool-o5fo8. gpu-run owns sync and unconditional deletion.
set -euo pipefail
cd /root/soup
export BEND_HOME=/opt/bend2
export SOUP_SOURCE_COMMIT
SOUP_SOURCE_COMMIT=$(cat build/gpu-source-commit.txt)
OUT=/root/runs/gpu-smoke-k12
[[ ! -e "$OUT/metrics.csv" ]] || { echo 'Refuse to overwrite a previous GPU timing run' >&2; exit 2; }
mkdir -p "$OUT"
date -u +%FT%TZ > "$OUT/started-at.txt"
printf '%s\n' "$SOUP_SOURCE_COMMIT" > "$OUT/source-commit.txt"
cp /var/lib/soup/READY "$OUT/vm-ready.txt"
if timeout --kill-after=30 480 scripts/build-native.sh --out build/native/soup > "$OUT/native-build.log" 2>&1; then
  ENGINE="/root/soup/build/native/soup --threads 8"
  LANE=native-cpu
  cp build/native/soup.build-info.txt "$OUT/"
else
  echo 'Native build failed; explicitly using JS on the VM to validate the pipeline.' | tee "$OUT/fallback.txt"
  for log in build/native/soup.check.log build/native/soup.build.log; do
    [[ ! -f "$log" ]] || cp "$log" "$OUT/"
  done
  bun "$BEND_HOME/bend2/main.ts" src/soup.bend -o build/gpu-smoke-soup.js
  ENGINE="bun /root/soup/build/gpu-smoke-soup.js"
  LANE=js-fallback
fi
printf '%s\n' "$LANE" > "$OUT/lane.txt"
# Only these new GPU outputs are written; no calibration directory is copied to this VM.
timeout --kill-after=15 120 $ENGINE 8 10 1 256 2 > "$OUT/smoke-256x10.csv"
python3 - "$OUT/smoke-256x10.csv" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1])))
assert len(rows)==10,len(rows)
w=256*1048576
for r in rows:
    assert int(r['wealth'])==w+int(r['trader_pnl'])-int(r['used'])-int(r['burned']),r
    w=int(r['wealth'])
print('256x10 smoke: PASS; exact ledger on all epochs')
PY
SOUP_CMD="$ENGINE" timeout --kill-after=30 600 scripts/run.sh --seed 1 --K 4096 --F 4096 --fuel 256 --epochs 100 --chunk 100 --out "$OUT" | tee "$OUT/run.log"
python3 - "$OUT" <<'PY'
import csv,json,pathlib,sys
p=pathlib.Path(sys.argv[1]); rows=list(csv.DictReader((p/'metrics.csv').open()))
assert len(rows)==100,len(rows)
w=4096*1048576
for r in rows:
    assert int(r['wealth'])==w+int(r['trader_pnl'])-int(r['used'])-int(r['burned']),r
    w=int(r['wealth'])
t=list(csv.DictReader((p/'timing.csv').open()))
seconds=sum(float(r['wall_s']) for r in t)
ops=sum(int(r['ops']) for r in rows)
assert ops==sum(int(r['instructions']) for r in t)
summary=dict(lane=(p/'lane.txt').read_text().strip(),epochs=len(rows),cells=4096,questions=4096,fuel=256,
             wall_s=seconds,instructions=ops,instructions_per_s=ops/seconds,
             js_reference_instructions_per_s=1080000,ratio_to_js_reference=(ops/seconds)/1080000,
             ledger='PASS',source_commit=(p/'source-commit.txt').read_text().strip())
(p/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
PY
date -u +%FT%TZ > "$OUT/finished-at.txt"
