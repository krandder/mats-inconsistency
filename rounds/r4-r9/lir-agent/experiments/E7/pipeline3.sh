#!/bin/bash
# E7 pipeline: wave 1 (warm starts + Gate 0) -> wave 2 (RL) -> wave 3 (analysis) -> aggregate (CPU).
set -u
E=/home/agent/workspaces/eurisko-study/lir-agent/experiments/E7
T=/tmp/lir-agent/e7
P=$T/pipeline.log
log() { echo "$(date -u +%FT%TZ) $*" >> $P; }
log "pipeline3: waiting for wave 1 launchers started by pipeline.sh"
until grep -q "finished\|NOT RUN" $T/launch-w1a.log 2>/dev/null && grep -q "finished\|NOT RUN" $T/launch-w1b.log 2>/dev/null; do sleep 60; done
n=$(ls $T/ckpt/warm-*/pre_rl.pt 2>/dev/null | wc -l); log "wave 1 done, warm ckpts=$n"
[ "$n" -ge 15 ] || { log "STOP: missing warm checkpoints"; exit 1; }
cd /tmp/lir-agent/e7 && source /home/agent/workspaces/eurisko-study/lir-agent/harness/scripts/env.sh
deg=$(python3 - <<'PY'
import json,glob,numpy as np
n=0
for f in glob.glob('/home/agent/workspaces/eurisko-study/lir-agent/experiments/E7/an/insitu_P-all_s*_pre_rl.json'):
    y=np.array(json.load(open(f))['y']); n+= (np.bincount(y).max()/len(y))>0.9
print(n)
PY
)
log "degenerate warm policies (mode>0.9): $deg"
[ "$deg" -lt 3 ] || { log "STOP: warm policies degenerate"; exit 1; }
log "wave 2 start"
$E/launch.sh w2a & sleep 30; $E/launch.sh w2b & sleep 30; $E/launch.sh w2c & wait
mkdir -p $T/ck3
for L in P-all P-role B0; do for s in 0 1 2 3 4; do
  d=$T/ck3/$L-s$s; mkdir -p $d
  ln -sfn $T/ckpt/warm-$L-s$s/pre_rl.pt $d/pre_rl.pt
  for t in rl30 post_rl; do [ -f $T/ckpt/$L-s$s/$t.pt ] && ln -sfn $T/ckpt/$L-s$s/$t.pt $d/$t.pt; done
done; done
n=$(ls $T/ck3/*/post_rl.pt 2>/dev/null | wc -l); log "wave 2 done, post_rl ckpts=$n"
log "wave 3 start"
$E/launch.sh w3a & sleep 30; $E/launch.sh w3b & wait
log "wave 3 done; aggregate"
PYTHONPATH=$PYTHONPATH:$T/pylib-mpl python3 $E/aggregate.py --an $E/an --runs $E/runs --out $E/results > $T/aggregate.log 2>&1
log "pipeline end rc=$?"
