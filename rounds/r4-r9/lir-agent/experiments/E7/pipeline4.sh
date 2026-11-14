#!/bin/bash
# E7 pipeline after wave 2 launchers are running: wait for them, then wave 3 (only if analysis jobs are
# missing), then aggregate on CPU.
set -u
E=/home/agent/workspaces/eurisko-study/lir-agent/experiments/E7
T=/tmp/lir-agent/e7
P=$T/pipeline.log
log() { echo "$(date -u +%FT%TZ) $*" >> $P; }
log "pipeline4: waiting for wave 2 launchers"
for m in w2a w2b w2c; do
  until grep -q "finished\|NOT RUN" $T/launch-$m.log 2>/dev/null; do sleep 60; done
done
mkdir -p $T/ck3
for L in P-all P-role B0; do for s in 0 1 2 3 4; do
  d=$T/ck3/$L-s$s; mkdir -p $d
  ln -sfn $T/ckpt/warm-$L-s$s/pre_rl.pt $d/pre_rl.pt
  for t in rl30 post_rl; do [ -f $T/ckpt/$L-s$s/$t.pt ] && ln -sfn $T/ckpt/$L-s$s/$t.pt $d/$t.pt; done
done; done
n=$(ls $T/ck3/*/post_rl.pt 2>/dev/null | wc -l); log "wave 2 done, post_rl ckpts=$n"
miss=0
for L in P-all P-role B0; do for s in 0 1 2 3 4; do for t in pre_rl rl30 post_rl; do
  [ -e $T/ck3/$L-s$s/$t.pt ] && [ ! -f $E/an/insitu_${L}_s${s}_$t.json ] && miss=$((miss+1))
done; done; done
for L in P-all B0; do for s in 0 1 2 3 4; do for t in pre_rl post_rl; do
  [ -e $T/ck3/$L-s$s/$t.pt ] && [ ! -f $E/an/fresh_${L}_s${s}_$t.json ] && miss=$((miss+1))
done; done; done
log "analysis jobs missing after wave 2: $miss"
if [ $miss -gt 0 ]; then log "wave 3 start"; $E/launch.sh w3; log "wave 3 done"; fi
cd /home/agent/workspaces/eurisko-study/lir-agent/harness && source scripts/env.sh
PYTHONPATH=$PYTHONPATH:$T/pylib-mpl python3 $E/aggregate.py --an $E/an --runs $E/runs --out $E/results > $T/aggregate.log 2>&1
log "pipeline end rc=$?"
