#!/bin/bash
# P1 job on trader-cat-p1-* (resume-safe: each step writes /root/out/markers/<step>.done).
# Two chains run concurrently on the one L4: T2m (mechanism) and mazes.
set -u
cd /root/job
PY=/opt/venv/bin/python
OUT=/root/out; DATA=/root/data
CK=$DATA/model_best.pth
mkdir -p $OUT/markers $OUT/logs $OUT/t2m $OUT/t1 $OUT/maze $DATA
touch $OUT/JOB_RUNNING
rm -f $OUT/JOB_DONE
ts() { date -u +%FT%TZ; }
step() {  # step NAME cmd...
  local name=$1; shift
  [[ -f $OUT/markers/$name.done ]] && return 0
  echo "$(ts) start $name"
  if "$@" >>$OUT/logs/$name.log 2>&1; then touch $OUT/markers/$name.done; echo "$(ts) done $name"
  else echo "$(ts) FAILED $name"; echo "$name" >>$OUT/FAILED_STEPS; return 1; fi
}
get_maze() {
  local n=$1 d=$DATA/maze_data_test_$1
  [[ -f $d/solutions.npy ]] && return 0
  curl -sSL --retry 3 -o $DATA/t$n.tgz https://www.cs.umd.edu/~tomg/download/Easy_to_Hard_Datav2/maze_data_test_$n.tar.gz &&
    tar -C $DATA -xzf $DATA/t$n.tgz && rm -f $DATA/t$n.tgz && ls $d
}
gate_pass() { $PY -c "import json,sys; sys.exit(0 if json.load(open('$1'))['pass'] else 1)"; }

sha256sum $CK $DATA/pilot_Bds.pt >$OUT/inputs.sha256
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >$OUT/gpu.txt
$PY -c "import torch;print(torch.__version__, torch.version.cuda)" >>$OUT/gpu.txt

# ------------------------------------------------------------------ data (CPU, in the background)
data_chain() {
  step dl13 get_maze 13
  step dl33 get_maze 33
  step dl59 get_maze 59
  step gen99 $PY -c "from data import write_generated; write_generated('$DATA', 99, 350, seed0=99000, procs=2, chunk=10)"
}

# ------------------------------------------------------------------ T2m chain
t2m_seed() {
  local s=$1 w=64 p=$OUT/t2m/t2m_bb_s$1.pt
  step bb$s env P1_GPU_FRAC=0.3 $PY train.py --task t2m --seed $s --width 64 --iters 10000 --curr_iters 3000 --out $p || return 1
  step bbgate$s env P1_GPU_FRAC=0.15 $PY train.py --task t2m --gate $p || return 1
  if ! gate_pass ${p%.pt}.gate.json; then
    p=$OUT/t2m/t2m_bb_s${s}_w128.pt
    step bb${s}w128 env P1_GPU_FRAC=0.33 $PY train.py --task t2m --seed $s --width 128 --iters 10000 --curr_iters 3000 --out $p || return 1
    step bbgate${s}w128 env P1_GPU_FRAC=0.15 $PY train.py --task t2m --gate $p || return 1
    gate_pass ${p%.pt}.gate.json || echo "backbone seed $s failed the gate after retry" >>$OUT/VALIDITY
  fi
  step t2m_eval$s env P1_GPU_FRAC=0.2 $PY eval.py t2m --bb $p --bseed $s --out $OUT/t2m
}
t2m_chain() {
  step t1smoke env P1_GPU_FRAC=0.15 $PY eval.py t1 --out $OUT/t1 --pilot $DATA/pilot_Bds.pt --head_batches 40 --bseed 0 &
  t2m_seed 0 &
  t2m_seed 1 &
  wait
}

# ------------------------------------------------------------------ maze chain
maze_chain() {
  local M="--out $OUT/maze --data $DATA --ckpt $CK"
  export P1_GPU_FRAC=0.6
  while [[ ! -f $OUT/markers/dl59.done ]]; do sleep 20; [[ -f $OUT/FAILED_STEPS ]] && grep -q '^dl' $OUT/FAILED_STEPS && return 1; done
  step mrepro $PY eval.py maze-repro $M
  for n in 13 33 59; do step mdevs$n $PY eval.py maze-devsimple --size $n $M; done
  step mgate $PY eval.py maze-gate $M || return 1
  step mheads $PY eval.py maze-heads $M || return 1
  local sizes="59 99"
  gate_pass $OUT/maze/headroom_gate.json && { sizes="13 19 25 33 59 99"; step dl19 get_maze 19; step dl25 get_maze 25; }
  while [[ ! -f $OUT/markers/gen99.done ]]; do sleep 20; [[ -f $OUT/FAILED_STEPS ]] && grep -q '^gen99' $OUT/FAILED_STEPS && { sizes=${sizes/99/}; break; }; done
  for n in $sizes; do step msize$n $PY eval.py maze-size --size $n --splits dev,eval $M; done
}

data_chain &
DPID=$!
t2m_chain &
TPID=$!
maze_chain &
MPID=$!
wait $TPID $MPID $DPID
echo "$(ts) all chains finished"
touch $OUT/JOB_DONE
rm -f $OUT/JOB_RUNNING
