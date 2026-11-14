#!/bin/bash
# E6 job scheduler: start train jobs and yoked replays (after X finishes) with at most MAXJ e6 python
# processes and at least MINMEM GB available memory before each start. Logs to /tmp/lir-agent/e6/sched.log
set -u
cd "$(dirname "$0")/../../harness"
source scripts/env.sh
E=../experiments/E6; CK=/tmp/lir-agent/e6/ckpt; MAXJ=${MAXJ:-5}; MINMEM=${MINMEM:-12}
TRAIN="c:0 c:2 d:2"
REPLAY="c:0 c:1 c:2 d:0 d:1 d:2"
declare -A started
nj() { pgrep -f "experiments/E6/e6run.py" -c 2>/dev/null | awk '{print int($1/2)}'; }   # timeout + python per job
mem() { free -g | awk '/Mem:/{print $7}'; }
start() { # kind X S
  local kind=$1 X=$2 S=$3 ARM; ARM=$([ "$X" = c ] && echo P || echo B0)
  if [ $kind = train ]; then
    nohup timeout 4h python3 $E/e6run.py --mode train --arm $ARM --seed $S --sft-steps 6 --sft-batch 8 --rl-steps 24 \
      --groups 2 --group-size 4 --eval-every 6 --threads 2 --micro-batch 4 --ckpt-dir $CK --out $E/runs/$X-s$S \
      > /tmp/lir-agent/e6/$X-s$S.log 2>&1 &
  else
    nohup timeout 4h python3 $E/e6run.py --mode replay --arm $ARM --seed $S --rl-steps 24 --eval-every 6 --threads 2 \
      --micro-batch 4 --ckpt-dir $CK --stream $E/runs/$X-s$S/stream --warm $CK/$X-s$S/warm.pt --out $E/runs/b$X-s$S \
      > /tmp/lir-agent/e6/b$X-s$S.log 2>&1 &
  fi
  started[$kind$X$S]=1; echo "$(date -u +%T) start $kind $X $S mem=$(mem) jobs=$(nj)" >> /tmp/lir-agent/e6/sched.log
}
while true; do
  pending=0
  for j in $TRAIN; do X=${j%:*}; S=${j#*:}; [ -n "${started[train$X$S]:-}" ] && continue; pending=1
    if [ $(nj) -lt $MAXJ ] && [ $(mem) -ge $MINMEM ]; then start train $X $S; sleep 90; fi; done
  for j in $REPLAY; do X=${j%:*}; S=${j#*:}; [ -n "${started[replay$X$S]:-}" ] && continue; pending=1
    [ -f $E/runs/$X-s$S/e6summary.json ] || continue
    if [ $(nj) -lt $MAXJ ] && [ $(mem) -ge $MINMEM ]; then start replay $X $S; sleep 90; fi; done
  [ $pending = 0 ] && break
  sleep 30
done
echo "$(date -u +%T) all started" >> /tmp/lir-agent/e6/sched.log
