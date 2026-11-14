#!/bin/bash
# usage: runjob.sh TAG SPEC  (one job; skips if done or claimed). Before its own job, drains extra.txt one line at a
# time under a lock (lines: TAG<TAB>SPEC[<TAB>MODE], MODE = e3 (default: E3 rerun), run (a main job) or exact), so the
# total number of processes stays at the xargs width. A main job is claimed (results/TAG.claimed) when it starts.
cd /home/agent/workspaces/mats/work/gate-entropy
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/home/agent/.venvs/aggext/bin/python
pop_extra() {
  ( flock 9; line=$(head -1 extra.txt 2>/dev/null); [ -n "$line" ] && sed -i '1d' extra.txt; echo "$line" ) 9>extra.lock
}
claim() {  # prints 1 if this caller claimed the tag
  ( flock 9; if [ -f "results/$1.done" ] || [ -f "results/$1.claimed" ]; then echo 0; else touch "results/$1.claimed"; echo 1; fi ) 9>extra.lock
}
runone() {  # tag spec
  if [ "$2" = "exact" ]; then
    nice -n 19 ionice -c3 $PY exact_check.py > logs/$1.log 2>&1 && touch results/$1.done
  else
    nice -n 19 ionice -c3 $PY run.py "$2" results/$1.json > logs/$1.log 2>&1 && touch results/$1.done
  fi
  echo "$(date -u +%H:%M:%S) $1 exit $?" >> logs/queue.log
}
while true; do
  line=$(pop_extra); [ -z "$line" ] && break
  etag=$(echo "$line" | cut -f1); espec=$(echo "$line" | cut -f2); emode=$(echo "$line" | cut -f3)
  if [ "$emode" = "run" ]; then
    [ "$(claim $etag)" = "1" ] && runone "$etag" "$espec"
  else
    nice -n 19 ionice -c3 $PY e3_rerun.py "$espec" "$etag" > logs/e3_$etag.log 2>&1
    echo "$(date -u +%H:%M:%S) e3rerun $etag exit $?" >> logs/queue.log
  fi
done
tag="$1"; spec="$2"
[ "$(claim $tag)" = "1" ] || exit 0
runone "$tag" "$spec"
