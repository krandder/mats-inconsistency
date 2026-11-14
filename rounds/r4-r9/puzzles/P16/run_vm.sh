#!/bin/bash
# On-VM driver: PLAN=<name> bash run_vm.sh. Every step is skipped when its marker exists (resume after preemption);
# a failed step is recorded in FAILED_STEPS and the plan continues; DONE_<plan> is written at the end.
set -u
cd /root/job
export PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 P16_OUT=/root/out
O=/root/out; mkdir -p $O/markers $O/logs $O/runs
PLAN=${PLAN:-prep}
hb() { while true; do touch $O/HEARTBEAT; sleep 60; done; }
hb & HB=$!; disown $HB   # not a job: a bare `wait` in a plan must not wait for the heartbeat
trap 'kill $HB 2>/dev/null' EXIT
step() {  # step NAME cmd...
  local n=$1; shift
  [[ -f $O/markers/$n ]] && { echo "skip $n"; return 0; }
  echo "$(date -u +%FT%TZ) start $n: $*"
  if "$@" >>$O/logs/$n.log 2>&1; then touch $O/markers/$n; echo "$(date -u +%FT%TZ) ok $n"
  else echo "$(date -u +%FT%TZ) FAIL $n"; echo "$PLAN $n $(date -u +%FT%TZ)" >>$O/FAILED_STEPS; return 1; fi
}
ev() { python eval.py "$@"; }
tuned() { python -c "import json,sys; print(json.dumps(json.load(open('$O/tuned.json'))['$1']))"; }

tune_c4() {
  for a in 1 2 4 8; do for r in 0.25 0.5; do
    step tune_C4_a${a}_r${r} ev run --split tune --arm C4 --params "{\"alpha\": $a, \"rho\": $r, \"tag\": \"a${a}_r${r}\"}"
  done; done
  step tune_pick1 python analyze.py tune
  tune_c5
}

tune_c5() {
  C4=$(tuned C4)
  for s in 0.5 1 2; do
    step tune_C5_s$s ev run --split tune --arm C5 --params "$(python -c "import json; d=json.loads('$C4'); d.update(scale=$s, tag='s$s'); print(json.dumps(d))")"
  done
}

plan_prep() {
  step design_tune ev design --split tune --n_per 21
  step design_pilot ev design --split pilot --n_per 42
  # E1 and L on the pilot run beside the tuning (python-overhead bound): one step per type (preemption granularity),
  # two L chains and two E1 chains in parallel
  pl() { for t in "$@"; do step pilot_L_$t ev run --split pilot --arm L --params "{\"types\": [\"$t\"], \"tag\": \"$t\"}"; done; }
  pe() { for t in "$@"; do step pilot_E1_$t ev run --split pilot --arm E1 --params "{\"types\": [\"$t\"], \"per\": 8, \"tag\": \"$t\"}"; done; }
  pl HP1 HN1 HP2 HN2 & pl HP3a HN3 HP3b HN4 &
  pe HP1 HN1 HP2 HN2 & pe HP3a HN3 HP3b HN4 &
  tune_c4
  for arm in S1 S2; do for lr in 1e-3 3e-3 1e-2; do for E in 10 20; do
    step tune_${arm}_lr${lr}_E$E ev run --split tune --arm $arm --params "{\"lr\": $lr, \"E_steps\": $E, \"tag\": \"lr${lr}_E$E\"}"
  done; done; done
  step tune_pick2 python analyze.py tune
  step pilot_C4 ev run --split pilot --arm C4 --params "$(tuned C4)"
  step pilot_C5 ev run --split pilot --arm C5 --params "$(tuned C5)"
  wait
  step gate python analyze.py gate
}

plan_prepfix() {
  tune_c4
  step pilot_C4 ev run --split pilot --arm C4 --params "$(tuned C4)"
  step pilot_C5 ev run --split pilot --arm C5 --params "$(tuned C5)"
}

# Stage 1 (after the gate addendum). Two VMs; every step resumes from its marker (training from its checkpoint).
plan_trainA() {  # FR x3 + FR-small, their evals, the adversarial slice; E1 on test beside the training
  step design_test ev design --split test --n_per 100
  te1() { local per=$1; shift; for t in "$@"; do
      step test_E1_$t ev run --split test --arm E1 --params "{\"types\": [\"$t\"], \"per\": $per, \"tag\": \"$t\"}"; done; }
  ( te1 12 HP1 HN1 HP2 HN2; te1 5 T0 T1 T2 ) &
  ( te1 12 HP3a HN3 HP3b HN4; te1 5 T3a T3b T4 T5 T6 ) &
  for r in FR_s0 FR_s1 FR_s2 FR-small_s0; do
    local arm=${r%_s*} s=${r##*_s}
    step train_$r python train.py --arm $arm --seed $s --fmeta 1e16 --out $O/runs
    for sp in test tune; do step ${sp}_$r ev run --split $sp --arm $arm --run $r; done
    if [[ $r == FR_s0 ]]; then step adv python adv.py --run FR_s0; fi
  done
  wait
}

plan_trainB() {  # FR-direct x2 and their evals; L and every baseline on test and tune beside the training
  step design_test ev design --split test --n_per 100
  tl() { for t in "$@"; do step test_L_$t ev run --split test --arm L --params "{\"types\": [\"$t\"], \"tag\": \"$t\"}"; done; }
  tl HP1 HN1 HP2 HN2 &
  ( tl HP3a HN3 HP3b HN4
    for sp in test tune; do
      step ${sp}_C1 ev run --split $sp --arm C1
      for a in C4 C5 S1 S2; do step ${sp}_$a ev run --split $sp --arm $a --params "$(tuned $a)"; done
    done
    step test_CONS ev run --split test --arm CONS ) &
  for s in 0 1; do
    step train_FR-direct_s$s python train.py --arm FR-direct --seed $s --fmeta 1e16 --out $O/runs
    for sp in test tune; do step ${sp}_FR-direct_s$s ev run --split $sp --arm FR-direct --run FR-direct_s$s; done
  done
  wait
}

plan_trainB2() {  # helper chain, started by hand on the trainB VM once its first L chain is free: trainB's own
  # baseline steps in reverse order (same step names and markers, so trainB skips whatever this chain has finished)
  step test_CONS ev run --split test --arm CONS
  for a in S2 S1 C5 C4; do step tune_$a ev run --split tune --arm $a --params "$(tuned $a)"; done
  step tune_C1 ev run --split tune --arm C1
  for a in S2 S1; do step test_$a ev run --split test --arm $a --params "$(tuned $a)"; done
}

plan_trainX() {  # EXPLORATORY, not pre-registered (added 2026-09-30 00:20Z after FR_s0 missed the ID AUROC precondition):
  # FR seed 0 at 4x F_meta (4e16), kept in its own output tree ($O/x) so the pre-registered analysis never reads it
  local X=$O/x; mkdir -p $X/eval $X/runs
  cp -n $O/eval/design_test.json $O/eval/design_tune.json $X/eval/; cp -n $O/tuned.json $X/
  step x_train_FR4e16_s0 python train.py --arm FR --seed 0 --fmeta 4e16 --out $X/runs
  for sp in test tune; do step x_${sp}_FR4e16_s0 env P16_OUT=$X python eval.py run --split $sp --arm FR --run FR_s0; done
}

plan_smoke() {
  step smoke_design ev design --split pilot --n_per 1
}

# E1 needs tabpfn (not in the VM image; PREREG pins 9.0.0). Install and fetch the V2 weights once, before parallel chains.
if [[ $PLAN == prep || $PLAN == trainA ]]; then
  python -c 'import tabpfn' 2>/dev/null || pip install -q tabpfn==9.0.0 >>$O/logs/install.log 2>&1
  python - >>$O/logs/install.log 2>&1 <<'PY'
import numpy as np
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
c = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device='cuda')
X = np.random.rand(20, 3).astype('float32'); y = (X[:, 0] > .5).astype(int)
c.fit(X, y); print('tabpfn ok', c.predict_proba(X[:2]).round(3).tolist())
PY
fi
plan_$PLAN
wait
touch $O/DONE_$PLAN
echo "$(date -u +%FT%TZ) plan $PLAN done"
