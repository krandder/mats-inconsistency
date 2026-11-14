#!/usr/bin/env bash
# E69 (amendment 77): wait until the GPU retry script (PID) has exited, find the run id whose VM ran, copy its e69_* runs into runs/e69/,
# then queue one job per net into the analysis queue /tmp/s7q: e48_eval_w1.py (through a one-entry symlink directory) and e69_heads.py.
# usage: e69_watch.sh RETRY_PID
cd /home/agent/workspaces/mats/work/cm-interp-B
PID=$1; LOG=results/e69_collect.log; PY=/home/agent/.venvs/aggext/bin/python
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
RID=$(grep -o "attempt [0-9]* (cmb-e69-[0-9]*): VM was created" GPU-LOG.md | tail -1 | grep -o "cmb-e69-[0-9]*")
echo "$(date -u +%TZ) retry script exited; rid=$RID" >> $LOG
[ -n "$RID" ] || exit 1
SRC=/tmp/l4runs/$RID/runs/$RID
mkdir -p runs/e69 results/e69
for d in "$SRC"/e69_*; do
  [ -d "$d" ] || continue
  n=$(basename "$d"); [ -e runs/e69/$n ] && continue
  cp -r "$d" runs/e69/$n; [ -f "$SRC/$n.log" ] && cp "$SRC/$n.log" runs/e69/$n.log
done
cp "$SRC/jobs.txt" runs/jobs_$RID.txt 2>/dev/null
echo "$(date -u +%TZ) collected $(ls -d runs/e69/*/ | wc -l) runs, $(ls runs/e69/*/meta.json 2>/dev/null | wc -l) finished" >> $LOG
for mp in runs/e69/*/meta.json; do
  run=$(dirname "$mp"); tag=$(basename "$run"); one=/tmp/s7q/one/$tag
  mkdir -p "$one"; [ -e "$one/$tag" ] || ln -s "$PWD/$run" "$one/$tag"
  echo "$PY e48_eval_w1.py $one results/e69 0 1 >> results/e69_eval.log 2>&1 && $PY e69_heads.py $run >> results/e69_heads.log 2>&1" > /tmp/s7q/pending/c69_$tag
done
echo "$(date -u +%TZ) queued evaluations" >> $LOG
