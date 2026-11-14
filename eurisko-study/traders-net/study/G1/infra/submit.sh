#!/bin/bash
# Submit a GPU job to the traders-net queue.
# Usage: infra/submit.sh --ws WS --name NAME --src DIR --hours H [--results-to DIR] [--sync-include 'PAT ...']
#   DIR must contain run.sh. It is copied (max 300 MB, no .git/__pycache__) to queue/pending/<ws>-<name>/.
#   On the VM: cwd = a copy of DIR, $OUT=/root/out, python = /opt/tn/bin/python (torch+CUDA, numpy,
#   scipy, pandas, matplotlib). run.sh is killed after H hours (H <= 5.5). Files in $OUT matching
#   *.csv *.json *.png *.md *.txt *.log (+ --sync-include patterns, each file <= 100 MB) are synced every
#   ~5 min and at the end to --results-to (default queue/<state>/<job>/out), which must be under
#   traders-net/study/<WS>/ or /tmp/traders-net/.
# Prints the job id. Dispatch happens within ~2 min (cron) or at once with infra/queue-runner.sh --once.
set -eu
source "$(dirname "$0")/lib.sh"
WS=""; NAME=""; SRC=""; HOURS=""; RTO=""; SINC=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ws) WS=$2; shift ;; --name) NAME=$2; shift ;; --src) SRC=$2; shift ;; --hours) HOURS=$2; shift ;;
    --results-to) RTO=$2; shift ;; --sync-include) SINC=$2; shift ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown flag $1" >&2; exit 2 ;;
  esac; shift
done
[[ $WS =~ ^[A-Z][A-Z0-9]{1,3}$ ]] || { echo "--ws like M7" >&2; exit 2; }
JOB="${WS,,}-$NAME"; valid_id "$JOB" || { echo "job id $JOB must match ^[a-z][a-z0-9-]{0,30}$" >&2; exit 2; }
[[ -f $SRC/run.sh ]] || { echo "$SRC/run.sh missing" >&2; exit 2; }
[[ $HOURS =~ ^[0-9]+([.][0-9]+)?$ ]] && awk -v h="$HOURS" 'BEGIN{exit !(h>0 && h<=5.5)}' || { echo "--hours in (0, 5.5]" >&2; exit 2; }
STUDY=$(dirname "$G1_DIR")
if [[ -n $RTO ]]; then
  RTO=$(realpath -m "$RTO")
  [[ $RTO == "$STUDY/$WS/"* || $RTO == /tmp/traders-net/* ]] || { echo "--results-to must be under $STUDY/$WS/ or /tmp/traders-net/" >&2; exit 2; }
fi
for s in pending running done failed refused; do
  [[ ! -e $QUEUE/$s/$JOB ]] || { echo "job $JOB already exists in queue/$s (pick another --name)" >&2; exit 2; }
done
kb=$(du -sk --exclude=.git --exclude=__pycache__ "$SRC" | cut -f1)
(( kb <= 300*1024 )) || { echo "$SRC is ${kb} KB > 300 MB; put data on the VM from run.sh instead" >&2; exit 2; }
R=$(reserve_usd "$HOURS")
echo "worst-case reserve \$$R; ledger now \$$(spend_usd) of \$$BUDGET_USD; $WS \$$(spend_usd "$WS") of \$$(ws_cap "$WS")" >&2
tmp="$QUEUE/.incoming-$JOB-$$"; mkdir -p "$tmp"
rsync -a --exclude=.git --exclude=__pycache__ --exclude=/out "$SRC/" "$tmp/"
{ echo "WS=$WS"; echo "HOURS=$HOURS"; [[ -z $RTO ]] || echo "RESULTS_TO=$RTO"; [[ -z $SINC ]] || echo "SYNC_INCLUDE=$SINC"
  echo "SUBMITTED=$(now_utc)"; echo "SRC=$(realpath "$SRC")"; } >"$tmp/job.env"
mv "$tmp" "$QUEUE/pending/$JOB"
echo "$JOB"
