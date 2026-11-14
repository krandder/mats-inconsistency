#!/bin/bash
# Phase-3 runner on a trader-mresnet VM (the ablation and the Bansal baseline, RESULTS.md section 17).
# p3/jobs.txt: training jobs, one per line "<name> <econ2|dtr> <args...>", 3 at a time;
# p3/reeval.txt: then the L3200 re-evaluations, "<name> reeval --ckpt <path>";
# p3/solo.txt (optional): then jobs run one at a time. Kinds: econ2, dtr, reeval, py (python3 <args>).
# Safe to rerun at any time: finished jobs have p3/<name>.done; training resumes from its
# <name>.ckpt.pt (written every 10 minutes); evals resume per (length, order) block.
# The VM's startup-script reruns this after a spot preemption and restart.
cd ~
mkdir -p p3
[ -f p3/p3.status ] && grep -q P3_DONE p3/p3.status && exit 0
run() { n=$1; s=$2; shift 2; [ -f p3/$n.done ] && return 0
  if [ "$s" = econ2 ]; then L=""; [ -f p3/$n.pt ] && L="--load p3/$n"
    python3 econ2.py "$@" $L --out p3/$n >> p3/$n.log 2>&1 < /dev/null
  elif [ "$s" = dtr ]; then python3 dtrecall.py "$@" --out p3/$n >> p3/$n.log 2>&1 < /dev/null
  elif [ "$s" = py ]; then python3 "$@" >> p3/$n.log 2>&1 < /dev/null
  else python3 reeval.py "$@" --out p3/$n.json >> p3/$n.log 2>&1 < /dev/null; fi && touch p3/$n.done; }
export -f run
echo "start $(date -u +%FT%TZ)" >> p3/p3.status
grep -v '^#' p3/jobs.txt | xargs -P 3 -L 1 bash -c 'run "$@"' _
grep -v '^#' p3/jobs.txt | while read -r l; do run $l; done      # retry alone any that failed
grep -v '^#' p3/reeval.txt | xargs -P 3 -L 1 bash -c 'run "$@"' _
grep -v '^#' p3/reeval.txt | while read -r l; do run $l; done
touch p3/solo.txt; grep -v '^#' p3/solo.txt | while read -r l; do run $l; done       # alone on the GPU (memory timing)
n=$(cat p3/jobs.txt p3/reeval.txt p3/solo.txt | grep -v '^#' | awk '{print $1}' | while read j; do [ -f p3/$j.done ] || echo $j; done | wc -l)
if [ "$n" = 0 ]; then echo "P3_DONE $(date -u +%FT%TZ)" >> p3/p3.status; else echo "P3_INCOMPLETE $n $(date -u +%FT%TZ)" >> p3/p3.status; fi
