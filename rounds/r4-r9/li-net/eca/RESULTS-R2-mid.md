## 3. R2a: a 1-layer network (depth bottleneck)

IMG: results/r2/kl-vs-M-r2a-r110.png

IMG: results/r2/kl-vs-M-r2a-r90.png

- **The gap is now large, but the one-step law is not learned either.** One-pass KL on validation: rule 110 F 0.34, B1 0.45,
  B2 0.26 nats; rule 90 F 0.10, B1 0.36, B2 0.049 (4-layer round 1: 3-seed means 0.006-0.080). But the one-step check S1 (the query's three
  parents known) is 0.037 (rule 110) and 0.077 (rule 90), above the 0.025 bar (4 layers: 3-seed means 0.008): one layer cannot even compute the local rule
  reliably, rule 90's XOR least of all. **R2-D2a FAILS for both rules** (on S1; the long-range half passes).
- **R2-D2b FAILS for both rules.** Rule 110, under sw_mix at M = 16: stiffness-first chains cut KL by 24% on F (bar 25%;
  they beat random order by 0.07 nats, CI above 0), 29% on B1 and 21% on B2 (CIs include 0, and random order does as well);
  under the plain mixture stiffness-first and confidence-first chains gain nothing (F +3%, B1 +7%, B2 -22%). Rule 90:
  stiffness-first and confidence-first chains gain at most 11% (CIs include 0); random order with the mixture gains 24% on B2
  (CI above 0) and nothing elsewhere. Random order is the best or tied-best selector on every
  bridge family, as in round 1.
- Reading (an interpretation): with a 1-layer network the conditionals that search chains through are themselves less
  accurate (S1 KL 0.04-0.08 vs about 0.008 at 4 layers), which limits what routing through them can gain. Stiffness still
  ranks pairs by information gain (Spearman 0.82 / 0.88, AUC 0.88 / 0.86), and on rule-110 F under sw_mix stiffness-first
  does beat random order (KL difference 0.066 [0.049, 0.082] at M = 16).
- The fitted sw_mix weights use stiffness mildly for rule 110 (eta = 0.5 in all 9 rule-110 fits at M = 16, lam 6-56) and
  little for rule 90 (eta = 0 in 6 of 9, 0.5-1 in the other 3).

## 4. R2b: keys without h_i, and VoI over uncertain cells (rule 110, 4 layers)

IMG: results/r2/kl-vs-M-r2b-r110.png

- **R2-D3a PASSES as pre-registered, but the pass is mostly by construction (audit H1).** cos(k_fwd, k_bwd) = -0.10 on B1
  and 0.07 on B2 (round 1: 0.95 and 0.95; bar <= 0.35), redundant views 0.91 (bar >= 0.85), drift ratio 0.052 at m = 5
  (0.149 at m = 2; bar < 0.20); every seed passes on its own. But the v2 key changed three things at once (no h_i, a
  fingerprint restricted to the newly materialised cells S minus E_0, a LayerNorm), and forward and backward chains
  materialise disjoint cells, so their restricted fingerprints have disjoint supports. The audit recomputed the same trained
  key over all revealed cells: 0.73 (B1) and 0.81 (B2), which would fail; and two disjoint chains in the SAME direction also
  score about 0.08. The key separates disjoint supports, not directions, and dropping h_i alone does not fix it. The
  redundant-view half is computed over all of S, so the two halves of the rule use different supports.
- The forward + backward pool beats both single chains on B1 (0.089 vs 0.118 / 0.127) and B2 (0.090 vs 0.120 / 0.137), but
  round 1's keys (cosine 0.95) gave the same ordering, so this is not an effect of the new keys (audit L1).
- **R2-D3b FAILS.** VoI over the 6 most uncertain cells is better than VoI over the 6 stiffest on B1 under the mixture
  (M = 8: 0.099 -> 0.068, cut 0.31 [0.15, 0.47], meeting the 30% bar) and beats one pass by 25% [12%, 38%] there, short of
  the 35% bar; under sw_mix it beats one pass by 31% [22%, 40%] but improves on stiff-gated VoI by only 19% (CI includes 0);
  on B2 it does not beat stiff-gated VoI (cuts 0.16 and 0.00). Random order (B1 0.047 at M = 8 under the mixture) is still
  better than either VoI.
- Note: the parameter-free attention-vector cosine between forward and backward chains is 0 by construction (their new
  cells are disjoint); the learned key inherits most of that separation.

## 5. The anchored pool (sw_mix), all selectors (secondary readout)

The round-1 stiffness-weighted pool made one-pass beliefs worse (rule-110 F 0.025 -> 0.21 at M = 0). The anchored pool
cannot do that (at M = 0 it equals P0), and on the 4-layer rule-110 models of R2b it turns every selector into a gain over
one pass at M = 16 on every family except VoI-uncertain on F (-2%); 14 of the 20 selector x family cuts have paired CIs
above 0 (on F only confidence-first's): e.g. F: stiffness-first -43% (0.025 -> 0.014, CI includes 0), random -33% (CI
includes 0); B1: random -46%, VoI-uncertain -32%, stiffness-first -24%; B2: random -40%, stiffness
-32%; R: -13% to -26%. Under the plain mixture most of these are losses (Sec. 10 table). This readout was added at 05:25Z
after the round-2 results were read, so it is secondary. On these 4-layer models the fitted stiffness exponent is 0 in 89
of 90 fits (lam 10-32 at M = 16, i.e. 38-66% of the weight on P0), so the gain comes from an anchored uniform mixture of
the chains, not from stiffness weighting. The round-1 pool also differed in being log-linear and anchored at the
network's prior rather than P0, so no readout isolates which difference matters most (audit M2).

## 6. Cost and clean-up

R2_COST

## 7. Deviations

- R2b's selector set adds stiff and conf (cheap) to the voi/voiu/rand that the rules need; R2a's adds no VoI arm (not
  needed by R2-D2).
- Round-2 runs are T1 only, as specified; P2-Stiff tau = 5 in R2a only (R2b keeps tau = 1 for comparability with round 1).
- `ecanet.Net` now builds an extra (unused unless --key-mode v2) keyW2 layer; every shared parameter initialises
  identically, but the global RNG stream after initialisation (dropout masks) differs, so round-2 runs are new trainings.
- The sw_mix-vs-P0 table (Sec. 5) and its figure rows are a secondary readout added after the results were read.
- The rule-90 1-layer runs vary widely across seeds: seed 2 learned the one-step rule (S1 0.021, validation B1 0.05), seeds
  0-1 did not (S1 0.10-0.11, B1 0.51), so the rule-90 CIs are wide.

## 8. Files

`PREREG-R2.md`; `analyze_r2.py`; `results/r2/summary.json`, `results/r2/tables.md`, `results/r2/*.png`; raw per-run
files and checkpoints `results/raw/r2a/r{rule}-T1L1-s{seed}/` and `results/raw/r2b/r110-T1-s{seed}/` (`result.json`,
`test-estimates.csv`, `d1-pairs.csv`, `train-history.csv`, `model.pt`). Code: `ecanet.py` (v2 key), `ecarun.py` (`--layers`,
`--key-mode`, `--tau`, `--selectors`, `--save-model`, selector `voiu`, pool `sw`).

## 9. Audit (`AUDIT-R2.md`)

An independent pass (Opus 5.5, 05:26-05:56Z) re-derived all four verdicts and every table line from the raw files with its
own code (0 mismatches), confirmed that the code diff contains only the flagged features with training unchanged, that
sw_mix and the v2 key are fitted on outcomes only, that sw at M = 0 equals P0, and the GPU total. It found: the key
separation is by construction (H1, applied in Sec. 4 and the TL;DR); PREREG-R2's pre-run save came 15-23 s after the
launch commands (M1, stated in the header and in `PREREG-R2-ERRATA.md`; the 05:49Z header edit was reverted at 06:02Z); sw_mix on the 4-layer runs is an anchored uniform mixture (M2, applied in Sec. 5);
and five small wording items (L1-L5, applied in Sec. 3-7 and, for L4, in the TL;DR at 06:02Z). This section and the
corrections were written after the audit (05:49-06:02Z).

## 10. Full tables

R2_TABLES
