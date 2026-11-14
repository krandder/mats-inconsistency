#!/bin/bash
# on trader-mresnet-3 after run_grid.sh: evaluate any checkpoint without an eval (D_* after the
# ponder_tau index fix, P_s0 etc.)
cd ~
until grep -q GRID_DONE results/eval_all.log 2>/dev/null && grep -q RETRY_DONE results/retry.status 2>/dev/null && ! pgrep -f 'train[.]py' >/dev/null; do sleep 60; done
todo=$(for f in results/*_s?.pt; do [ -f ${f%.pt}.eval.json ] || echo $f; done)
[ -n "$todo" ] && python3 evaluate.py $todo > results/eval_rest.log 2>&1
echo EVALREST_DONE >> results/eval_rest.log
