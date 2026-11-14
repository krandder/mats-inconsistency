#!/bin/bash
# Item 1b, one seed (PREREG-1b.md): extract the official maze data once (flock, sha256-checked) -> selftest once ->
# train (BBPos, --norm 1, lambda 20, 12000 steps) -> val eval -> wait for the arm gate (gate.py 1b) -> test evals.
# Every step is skipped when its output exists.
S=$1; SEEDS=${2:-200-219}
PY=${PY:-/opt/venv/bin/python}; R=/root/out/runs1b; E=/root/out/eval1b; mkdir -p $R $E/val $E/test
export FU_BANSAL_DATA=/root/bansal
cd /root/job || exit 1
(
  flock 9
  if [[ ! -f /root/bansal/EXTRACTED ]]; then
    mkdir -p /root/bansal && (cd bansal_data && sha256sum -c SHA256SUMS) || { date > $E/PRE_FAILED; exit 1; }
    for f in bansal_data/*.tar.gz; do tar xzf $f -C /root/bansal || { date > $E/PRE_FAILED; exit 1; }; done
    date > /root/bansal/EXTRACTED
  fi
  if [[ ! -f $E/SELFTEST_OK && ! -f $E/PRE_FAILED ]]; then
    $PY bansal.py selftest > $E/selftest.log 2>&1 && date > $E/SELFTEST_OK || date > $E/PRE_FAILED
  fi
) 9>/root/bansal.lock
[[ -f $E/PRE_FAILED ]] && exit 1
CK=$R/bansal_s$S.pt
if [[ ! -f $CK ]]; then
  $PY bansal.py train --seed $S --lam 20 --norm 1 --steps 12000 --threads 1 --out $CK || { date > $E/TRAINFAILED_s$S; exit 1; }
fi
[[ -f $E/val/bansal_s$S.json ]] || $PY bansal.py eval --ckpt $CK --sets val --outdir $E/val --threads 1 || { date > $E/EVALFAILED_s$S; exit 1; }
$PY gate.py 1b --dir $E --seeds $SEEDS || exit 1
[[ -f $E/test/bansal_s$S.json ]] || $PY bansal.py eval --ckpt $CK --sets e2h_maze_13,e2h_maze_33,e2h_maze_59,e2h_maze_201 --outdir $E/test --threads 1 || exit 1
exit 0
