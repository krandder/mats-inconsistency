#!/usr/bin/env bash
# Runs on the L4 VM from /root/soup: phase 1 (jobs1.txt) then phase 2 (jobs2.txt, fine-tunes of phase-1 nets).
# Job line: NAME SCRIPT ARGS. sftrain.py jobs train with the on-device sampler (--tdata) and are evaluated on the
# test split; e4fa.py jobs train and evaluate likewise. PAR jobs at once; CUDA MPS on.
# Outputs: /root/runs/$RID/<name>/ and <name>.log
RID=$1; PAR=${2:-12}
PY=/root/venv/bin/python; OUT=/root/runs/$RID
mkdir -p $OUT
nvidia-smi -L; $PY -c "import torch, numpy, scipy; print(torch.__version__, numpy.__version__, torch.cuda.is_available())" || $PY -m pip install -q scipy
$PY -c "import sfgpu, sfmodels, sftrain, e4fa; print('imports ok')" || { echo IMPORT-FAIL; exit 3; }
export OMP_NUM_THREADS=1
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d && echo MPS-ON
run_phase() {
  local file=$1
  while read -r NAME SCRIPT ARGS; do
    [ -z "$NAME" ] && continue
    if [ -f "$OUT/$NAME/eval_test_model.json" ] || [ -f "$OUT/$NAME/eval_test.json" ]; then continue; fi
    while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 2; done
    ARGS=${ARGS//@OUT@/$OUT}
    if [ "$SCRIPT" = "sftrain.py" ]; then
      ( $PY sftrain.py train $ARGS --tdata --device cuda --out $OUT/$NAME && \
        $PY sftrain.py eval --run $OUT/$NAME --split test --device cuda ) > $OUT/$NAME.log 2>&1 &
    else
      ( $PY $SCRIPT train $ARGS --device cuda --out $OUT/$NAME && \
        $PY $SCRIPT eval --run $OUT/$NAME --split test --device cuda ) > $OUT/$NAME.log 2>&1 &
    fi
    sleep 0.3
  done < "$file"
  wait
}
date -u; run_phase jobs1.txt; echo PHASE1-DONE; date -u
run_phase jobs2.txt; echo PHASE2-DONE; date -u
echo quit | nvidia-cuda-mps-control
ls $OUT | wc -l
grep -l -i "error\|traceback" $OUT/*.log | head -20
echo SFX-DONE
