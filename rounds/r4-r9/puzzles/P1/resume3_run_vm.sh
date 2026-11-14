#!/bin/bash
# P1 second resume, after the spot preemption of trader-cat-p1-2 (15:32Z). /root/out = the p1-2 sync of 15:29Z
# (final backbones and gates done). T2m: t2m_run_vm.sh skips bb/gate by marker and runs t2m_eval{0,1}; head training,
# the maze TT fit and the maze pool chunks now write partial checkpoints (eval.py), so a further preemption loses
# minutes. Mazes: data for 59/99 again (fresh /root/data; new marker names), then msize59 and msize99 once
# /root/job/GO_MAZE exists (set after the local maze resume test passes). Name contains "run_vm.sh" for the watchdog.
set -u
cd /root/job
PY=/opt/venv/bin/python
OUT=/root/out; DATA=/root/data
CK=$DATA/model_best.pth
mkdir -p $OUT/markers $OUT/logs $OUT/maze $DATA
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
get_maze() {
  local n=$1 d=$DATA/maze_data_test_$1
  [[ -f $d/solutions.npy ]] && return 0
  curl -sSL --retry 3 -o $DATA/t$n.tgz https://www.cs.umd.edu/~tomg/download/Easy_to_Hard_Datav2/maze_data_test_$n.tar.gz &&
    tar -C $DATA -xzf $DATA/t$n.tgz && rm -f $DATA/t$n.tgz && ls $d
}
sha256sum $CK $DATA/pilot_Bds.pt >$OUT/inputs3.sha256
sha256sum /root/job/*.py /root/job/*.sh >$OUT/code3.sha256
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >$OUT/gpu3.txt

maze_chain() {
  local M="--out $OUT/maze --data $DATA --ckpt $CK"
  export P1_GPU_FRAC=0.6
  step dl59c get_maze 59 || return 1
  step gen99c $PY -c "from data import write_generated; write_generated('$DATA', 99, 350, seed0=99000, procs=2, chunk=10)" &
  local G=$!
  until [[ -f /root/job/GO_MAZE ]]; do sleep 20; done
  sha256sum /root/job/*.py >$OUT/code3_maze.sha256
  step msize59 $PY eval.py maze-size --size 59 --splits dev,eval $M
  wait $G
  [[ -f $OUT/markers/gen99c.done ]] || return 1
  step msize99 $PY eval.py maze-size --size 99 --splits dev,eval $M
}

bash t2m_run_vm.sh >>$OUT/t2m_run_vm.out 2>&1 &
TPID=$!
maze_chain &
MPID=$!
wait $TPID $MPID
kill $KEEP 2>/dev/null
echo "$(ts) resume3 chains finished"
touch $OUT/JOB_DONE
rm -f $OUT/JOB_RUNNING
