#!/bin/bash
# X1 worker 4: fresh-seed (4) reruns of M8, M9(c), K1 in a clean copy. See PREREG.md.
# Copy made 2026-09-28 12:1xZ: study/{S0/substrate,M8/w1,M8/w2,M9/w1,M9/w3,M6/w1,K1} -> /tmp/traders-net/X1w4/study/
# Only edits to the copy: /tmp output paths (sign_gate.py census npz, gated.py logits, run_w1.py TMP) and
# econ.py's runlog import path -> the copy's S0/substrate. Diff: /tmp/traders-net/X1w4/copy.diff (also X1/w4/copy.diff).
C=/tmp/traders-net/X1w4/study; L=/tmp/traders-net/X1w4/logs; mkdir -p $L
export PYTHONPATH=/home/agent/workspaces/eurisko-study/wealth-net/.pylib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export KQ_OUT=$C/M9/w3/results KQ_TMP=/tmp/traders-net/X1w4/M9w3
jobs() {
  echo "k1w3crp|cd $C/K1/w3 && python3 fof.py --rule crp --topo within --seed 4 --N 10000 --T 10000"
  echo "k1w3bh|cd $C/K1/w3 && python3 fof.py --rule bh --topo within --seed 4 --N 10000 --T 10000"
  for a in none pos1 neg1 shuf1; do echo "m8w1$a|cd $C/M8/w1 && python3 sign_gate.py $a 4"; done
  for a in absag random; do echo "m8w2$a|cd $C/M8/w2 && python3 gated.py $a 4"; done
  for a in S-on10k S-off10k C0-10k C0.1-10k; do echo "k1w2$a|cd $C/K1/w2 && python3 w2run.py $a 4"; done
  for c in "B 1 128" "B 64 2" "B 1 8" "B 64 8" "A 1 128" "A 64 2"; do set -- $c; echo "m9$1K$2hw$3|cd $C/M9/w3 && python3 run_kq.py $1 $2 $3 4"; done
  for n in on off; do echo "k1w1$n|python3 /home/agent/workspaces/eurisko-study/traders-net/study/X1/w4/k1_w1_one.py $n 4"; done
}
jobs | xargs -P 3 -d '\n' -I{} bash -c '
  j="{}"; name=${j%%|*}; cmd=${j#*|}
  [ -f '$L'/$name.done ] && exit 0
  while [ "$(cut -d" " -f1 /proc/loadavg | cut -d. -f1)" -gt 40 ]; do sleep 30; done
  echo "$(date -u +%T) START $name" >> '$L'/runner.log
  bash -c "$cmd" > '$L'/$name.log 2>&1; rc=$?
  echo "$(date -u +%T) END $name rc=$rc" >> '$L'/runner.log
  [ $rc = 0 ] && touch '$L'/$name.done'
