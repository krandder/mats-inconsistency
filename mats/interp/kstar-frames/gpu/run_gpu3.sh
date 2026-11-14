#!/usr/bin/env bash
# Runs on the L4 VM from /root/runs/$RID: the jobs of jobs_gpu3.txt one at a time (reruns of kstar2 jobs that failed
# with CUDA out-of-memory while several jobs shared the GPU). Line format: <log name>|<script and args>.
set -u
PY=${PY:-/root/venv/bin/python}
export PY KF_DEVICE=cuda OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out/runs out/logs
nvidia-smi -L
cd scratch
while IFS='|' read -r name cmd; do
  [ -z "$name" ] && continue
  $PY $cmd > ../out/logs/$name.log 2>&1
  echo "$name rc=$? $(date -u +%H:%M:%SZ)"
done < ../jobs_gpu3.txt
grep -l Traceback ../out/logs/*.log
du -sh ../out
echo RUN-GPU-DONE
