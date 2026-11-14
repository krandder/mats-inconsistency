#!/bin/bash
# P1 resume after the spot preemption of trader-cat-p1-1 (14:32Z): the remaining steps only.
# /root/out holds the p1-1 sync (markers, heads, partial backbones); /root/data is fresh, so the maze data for
# 59 and 99 is fetched/regenerated again (same URL, same generator seeds). Mazes: msize59, msize99. T2m: t2m_run_vm.sh
# (bb{s}v2 resumes from its .partial at it 3999). The file name contains "run_vm.sh" so the watchdog counts it as the job.
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
sha256sum $CK $DATA/pilot_Bds.pt >$OUT/inputs2.sha256
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >$OUT/gpu2.txt

maze_chain() {
  local M="--out $OUT/maze --data $DATA --ckpt $CK"
  export P1_GPU_FRAC=0.6
  step dl59r get_maze 59 || return 1
  step gen99r $PY -c "from data import write_generated; write_generated('$DATA', 99, 350, seed0=99000, procs=2, chunk=10)" &
  local G=$!
  step msize59 $PY eval.py maze-size --size 59 --splits dev,eval $M
  wait $G
  [[ -f $OUT/markers/gen99r.done ]] || return 1
  step msize99 $PY eval.py maze-size --size 99 --splits dev,eval $M
}

bash t2m_run_vm.sh >>$OUT/t2m_run_vm.out 2>&1 &
TPID=$!
maze_chain &
MPID=$!
wait $TPID $MPID
kill $KEEP 2>/dev/null
echo "$(ts) resume chains finished"
touch $OUT/JOB_DONE
rm -f $OUT/JOB_RUNNING
