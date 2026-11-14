#!/usr/bin/env bash
# Run ForecastWorld jobs on the L4 VM: each line of $1 is "cell tag arms..."; up to $2 jobs at once.
# Writes /root/runs/$RUN/cells/*.json and logs/*.log.
set -u
JOBS=$1; PAR=${2:-6}; RUN=$3; STEPS=${4:-6000}; export EXTRA=${5:-}
mkdir -p /root/runs/$RUN/cells /root/runs/$RUN/logs
nvidia-smi -L || true
grep -v '^#' "$JOBS" | grep -v '^\s*$' | xargs -P "$PAR" -L 1 bash -c '
  cell=$0; tag=$1; shift
  python scripts/${RUNNER:-forecast_world_run.py} --cells "$cell" --arms "$@" --tag "$tag" --steps '"$STEPS"' --threads 1 \
    ${EXTRA:-} --out /root/runs/'"$RUN"'/cells > /root/runs/'"$RUN"'/logs/${cell//:/_}_$tag.log 2>&1
  echo "JOB $cell $tag rc=$?"'
echo ALL-DONE
