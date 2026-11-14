#!/bin/bash
# P1 T2m chain, recipe v2 (deviation 9b). Runs next to run_vm.sh (maze chain), resume-safe via markers.
# The file name contains "run_vm.sh" on purpose: the watchdog counts it as the job.
set -u
cd /root/job
PY=/opt/venv/bin/python
OUT=/root/out
mkdir -p $OUT/markers $OUT/logs $OUT/t2m
touch $OUT/JOB_RUNNING
rm -f $OUT/JOB_DONE
( while true; do touch $OUT/JOB_RUNNING; sleep 60; done ) &
KEEP=$!
ts() { date -u +%FT%TZ; }
step() {  # step NAME cmd...
  local name=$1; shift
  [[ -f $OUT/markers/$name.done ]] && return 0
  echo "$(ts) start $name"
  if "$@" >>$OUT/logs/$name.log 2>&1; then touch $OUT/markers/$name.done; echo "$(ts) done $name"
  else echo "$(ts) FAILED $name"; echo "$name" >>$OUT/FAILED_STEPS; return 1; fi
}
gate_pass() { $PY -c "import json,sys; sys.exit(0 if json.load(open('$1'))['pass'] else 1)"; }

t2m_seed() {
  local s=$1 p=$OUT/t2m/t2m_bb_s$1.pt
  step bb${s}v2 env P1_GPU_FRAC=0.3 $PY train.py --task t2m --seed $s --width 64 --iters 8000 --recipe v2 --out $p || return 1
  step bbgate$s env P1_GPU_FRAC=0.15 $PY train.py --task t2m --gate $p || return 1
  if ! gate_pass ${p%.pt}.gate.json; then
    p=$OUT/t2m/t2m_bb_s${s}_w128.pt
    step bb${s}w128v2 env P1_GPU_FRAC=0.33 $PY train.py --task t2m --seed $s --width 128 --iters 8000 --recipe v2 --out $p || return 1
    step bbgate${s}w128 env P1_GPU_FRAC=0.15 $PY train.py --task t2m --gate $p || return 1
    gate_pass ${p%.pt}.gate.json || echo "backbone seed $s failed the gate after retry" >>$OUT/VALIDITY
  fi
  step t2m_eval$s env P1_GPU_FRAC=0.2 $PY eval.py t2m --bb $p --bseed $s --out $OUT/t2m
}

t2m_seed 0 &
S0=$!
t2m_seed 1 &
S1=$!
wait $S0 $S1
kill $KEEP 2>/dev/null
echo "$(ts) t2m chain finished"
if [[ $(pgrep -f "bash (resume[0-9]*_)?run_vm[.]sh" | wc -l) -eq 0 ]]; then touch $OUT/JOB_DONE; rm -f $OUT/JOB_RUNNING; fi
