#!/bin/bash
# P15 job on trader-b3-p15-* (resume-safe: each step writes /root/out/markers/<step>.done; each queue job
# skips when results/<id>.json exists and resumes from ckpt/<id>.pt).
# PLAN is a comma list of: setup | smoke | gates | G2b|G3|G4|G6 | g0 | g5 | g5pp | q:<name> | q1:<name> (NPAR=1) | mem |
# wait | end, each optionally prefixed bg: (background)
set -u
cd /root/job
PY=/opt/venv/bin/python
OUT=/root/out
NPAR=${NPAR:-4}
PLAN=${PLAN:-setup,smoke}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
mkdir -p $OUT/markers $OUT/logs $OUT/results $OUT/ckpt $OUT/gates /root/cache /root/ext/data
for d in results ckpt gates logs; do [[ -L /root/job/$d ]] || { rm -rf /root/job/$d; ln -s $OUT/$d /root/job/$d; }; done
[[ -L /root/job/cache ]] || { rm -rf /root/job/cache; ln -s /root/cache /root/job/cache; }
touch $OUT/JOB_RUNNING
rm -f $OUT/JOB_DONE
ts() { date -u +%FT%TZ; }
beat() { while sleep 60; do touch $OUT/HEARTBEAT; done; }
beat & BEAT=$!
step() {  # step NAME cmd...
  local name=$1; shift
  [[ -f $OUT/markers/$name.done ]] && return 0
  echo "$(ts) start $name"
  if "$@" >>$OUT/logs/$name.log 2>&1; then touch $OUT/markers/$name.done; echo "$(ts) done $name"
  else echo "$(ts) FAILED $name"; echo "$name" >>$OUT/FAILED_STEPS; return 1; fi
}
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >$OUT/gpu.txt
$PY -c "import torch;print(torch.__version__, torch.version.cuda)" >>$OUT/gpu.txt

fetch() (  # easy-to-hard data (Bansal et al.), for G0 (KS) and G5; subshell keeps the cwd
  cd /root/ext/data || exit 1
  local U=https://www.cs.umd.edu/~tomg/download/Easy_to_Hard_Datav2
  for f in prefix_sums_data maze_data_test_9 maze_data_test_59; do
    [[ -d $f ]] && continue
    curl -sSfL --retry 3 -o $f.tar.gz $U/$f.tar.gz && tar -xzf $f.tar.gz && rm -f $f.tar.gz || exit 1
  done
  ls -la /root/ext/data
)

run_one() {
  local s=$1
  case $s in
    setup) step setup fetch ;;
    smoke) step smoke $PY runq.py queue_smoke.json $NPAR ;;
    gates) step gates $PY gates.py G2b G3 G4 G6 ;;
    G2b|G3|G4|G6) step gate_$s $PY gates.py $s ;;
    g0) step g0 $PY g0.py ;;
    g5) step g5_ref $PY g5.py ref
        local pids=()
        step g5_pub $PY g5.py pub & pids+=($!)
        for sd in 0 1 2; do step g5_retrain_s$sd $PY g5.py retrain $sd & pids+=($!); sleep 2; done
        wait "${pids[@]}"
        step g5_ours $PY g5.py ours ;;
    g5pp) step g5_pubprefix $PY g5.py pubprefix ;;
    g5m) step g5_oursmaze $PY g5.py oursmaze ;;  # P-pub maze on our sets (prefix part runs with g5c)
    g5fix) for sd in 1 2; do step g5_retrainraw_s$sd $PY g5.py retrainraw $sd; done ;;  # uncentred ablation evals (hit CUDA OOM)
    g5c) local pids=()  # centred P-pub prefix retrains (Bansal's recipe), then P-pub prefix on our sets
        for sd in 0 1 2; do step g5_retrainc_s$sd $PY g5.py retrain $sd & pids+=($!); sleep 2; done
        wait "${pids[@]}"
        step g5_oursprefix $PY g5.py oursprefix ;;
    q:*) local n=${s#q:}; step q_$n $PY runq.py queue_$n.json $NPAR ;;
    q1:*) local n=${s#q1:}; step q_$n $PY runq.py queue_$n.json 1 ;;
    q2:*) local n=${s#q2:}; step q_$n $PY runq.py queue_$n.json 2 ;;  # NPAR=2 where G5 evals share the GPU
    mem) step mem $PY mem.py ;;
    wait) [[ ${#BG[@]} -gt 0 ]] && wait "${BG[@]}"; BG=() ;;
    end) : ;;
    *) echo "$(ts) unknown plan step $s" ;;
  esac
}

# PLAN items run in order; "bg:<item>" runs <item> in the background, "wait" waits for all of them
BG=()
IFS=, read -ra STEPS <<<"$PLAN"
for s in "${STEPS[@]}"; do
  if [[ $s == bg:* ]]; then run_one "${s#bg:}" & BG+=($!); else run_one "$s"; fi
done
[[ ${#BG[@]} -gt 0 ]] && wait "${BG[@]}"
echo "$(ts) plan $PLAN finished"
kill $BEAT 2>/dev/null
touch $OUT/JOB_DONE
rm -f $OUT/JOB_RUNNING
