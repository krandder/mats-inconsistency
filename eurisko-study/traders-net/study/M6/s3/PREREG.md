# M6 stress item 4 (worker s3): D families (delay, prove, expensive) for the surviving arms

Written 2026-09-28 ~09:45Z, before any run. Claim: `../STATUS.md` (s3 CLAIM-FIX). Siblings: s1 owns
multi-epoch + independent evidence, s2 owns 50% noise traders. Base-pass verdicts: `../M6.md`,
`../lead/M6-verdict.json`.

## Stream
S0 `li_families.make_li_stream(seed, T=20000)` (68 questions: D-delay 0/64/512 with twin and solo
questions, AND/OR over delay-0 bases, D-prove taut_or/contra/impl/ex3 revealed late by a
deductive process with staggered releases, D-expensive CA110/LFSR16 k in {64,256,1024} ahead).
Single prequential pass (`prequential.run_prequential`, B=4): each (position, question) is scored
at trade time and trained on once, when its label is revealed (t + delay). Seeds 1, 2, 3.

## One harness for every arm (`dfam.py`)
Trunk S0 `model.Net` d=32, L=1, H=4 up to the final LN (as w4's LI runs). Market maker l0 = S0
`Logistic`, trained on its own loss (Adam 1e-2), detached inside the layer. 16 trader heads per
question, w2's head: d_i = <y_q, rw_i[q]> + rc_i[q]*l0 + rb_i[q]. Net optimiser Adam 1e-3.
Arms (surviving base arms from `../lead/M6-verdict.json`, rebuilt in this harness):
- a      z = sum_i d_i (reference)
- b      z = l0 + sum_i d_i
- cg     p = sum_i k_i sigmoid(l0 + d_i), k = softmax(theta), theta [16]
- cq     same, theta [Q,16]
- fb01   z = l0 + sum_i 0.1 tanh(d_i / 0.1)
- fb05   z = l0 + sum_i 0.5 tanh(d_i / 0.5)
Kill controls (in-harness): wd = (a) with AdamW weight decay 0.1; boost = l0 + 0.1 sum_i d_i.
Early stopping is not defined in a single prequential pass (no epochs) and is not run here; s1
covers it in the multi-epoch regime. 1-pass/2-pass/fixed point on this stream already exist in
w4 (`../w4/results/li-*.json`); they are cited, not rerun (different head parameterisation, so
only within-harness comparisons are made). Committed heads (w5 Cm/Cw) are not rebuilt (their
harness is 20k-step base stream with a sequential settlement loop); noted as not covered.

## Metrics (trade-time prices; positions >= n_pos/2 for BCE unless stated)
- P1 D-learnable BCE: mean over families {delay0, delay64, delay512, and, or, exp64, exp256,
  exp1024} of family BCE.
- P2 pre-resolution D-prove violation: mean over kinds {taut_or, contra, impl, ex3} of the
  `li_families.violations(..., t)` "pre:" means (all live steps with no revealed history), as w4's M3.
  The maker's uninformed quote gives 0.500.
- P3 twin violation: mean |p_twin - p_base| over live steps (the twin's label equals its base's).
- Secondary: per-family BCE; twin vs solo BCE at D=64 and 512; post-resolution prove BCE;
  D-expensive excess over ln 2 and over the maker; AND/OR violations; maker's values.

## Stop rules
- D-stress failure for arm X in {b, cg, cq, fb01, fb05}: P1(X) >= P1(a) * 1.01 in 3/3 seeds
  (paired) AND no protection beyond the kill controls, where protection = P2 or P3 below
  min(wd, boost) in 3/3 paired seeds. Failure = "the arm does not survive the D stress".
- "Coherent before resolution" is claimed for an arm only if P2 <= 0.25 (a 50% cut against the
  maker's 0.5) in 3/3 seeds.
- Expected (planner's table, Q3): coherent on AND, not on OR or D; nothing trains a D-prove
  question before its reveal, so P2 stays near 0.5 for every arm.
Time cap: 3 h wall. Compute: one G1 VM job (CPU runs on its 8 vCPU, --hours 1.0) since this box is at
load ~34; fallback CPU here with at most 3 parallel jobs.

Compute note (09:52Z, before the real runs): a T=2400 smoke run takes 15 s here, so all 24 runs go on
this box's CPU, 3 at a time (`run_all.sh`); no G1 job. Smoke outputs in /tmp/traders-net/M6s3/smoke/ are not results.
