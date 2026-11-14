# E-CA round 2 pre-registration (eurisko, 2026-10-15, written 04:29-04:30Z, before the round-2 launches at 04:30:13Z)

Source: gemini's round-1 review `/home/agent/workspaces/mats/outbox/overnight-followups-round-1.md` (04:20Z), ranks 2 and 3
(E-CA-R2a, E-CA-R2b), assigned to eurisko by opus-mats (mail 3623, relaying Kelvin). Cap: $25 for round 2 (group
overnight-ca, run ids eca-*; guard at $105 overall). Round-1 design, splits, data, exact inference, families, training
protocol (early stopping on validation outcomes, patience 10 x 1000 steps), heads, chains (K = 16, M in {0,1,2,4,8,16}),
pools and rules are unchanged unless listed here (PREREG.md). Kelvin: "even negative results are fine"; every rule is
reported pass/fail with numbers; no retuning after test. Code: `ecanet.py`, `ecarun.py` (round-2 options are flags; with
the defaults the round-1 behaviour is unchanged, except that ecanet.Net now also builds an unused keyW2 layer).

## Changes (all evaluation-side unless stated)
1. **sw_mix pool** (both R2a and R2b): mu = (lam P0 + sum_r w_r q_r) / (lam + sum_r w_r), w_r = beta_int(Y | S_M^r)^eta,
   P0 = q(Y | E) (one pass). (lam, eta) fitted per (selector, M) on VALIDATION OUTCOME log loss over F, B1, B2, R by grid:
   eta in {0, 0.5, 1, 2}, lam in {0} U logspace(-3, 5, 33). At M = 0, mu = P0 exactly. (`ecarun.sw_pool`, `fit_sw`.)
2. **P2-Stiff temperature** tau = 5 in R2a (`--tau 5`); R2b keeps tau = 1 (round-1 value) so its stiff arm stays comparable.
3. **R2b key head** (`--key-mode v2`): k(i | S) = normalise(W2 LayerNorm(sum_j abar_{i->j} pos(j))), abar = token i's
   last-layer head-averaged attention restricted to the newly materialised cells S \ E_0 when the row has any (else to all
   of S), renormalised to sum to 1; no h_i. W2 (Linear d -> 16) is trained exactly as round 1's W_K (cal split, outcome log
   loss of the Gram-crowded pool on shared-core vs disjoint bags). Also logged: the parameter-free cosine between the
   attention vectors abar of two claims (D3a views; D3b forward vs backward chains). Note: for D3b the forward and backward
   chains materialise disjoint cells, so their restricted attention vectors have disjoint supports and cosine 0 by
   construction; the key cosine is the informative number.
4. **R2b uncertainty-gated VoI** (selector `voiu`): VoI among the 6 masked cells with the highest q(1 - q) (+1e-4 random
   tie-break), instead of the 6 stiffest (`voi`, kept and rerun on the same models for the paired comparison).
5. **R2a depth bottleneck**: `--layers 1` (training change: a 1-layer network; everything else as round 1).
6. model.pt saved per run and synced home (`--save-model 1`; driver now also syncs *.pt) for later rounds.

## Runs
- R2a: rules 110 and 90, T1 only, seeds 0, 1, 2, `--layers 1 --tau 5 --selectors stiff,conf,rand --save-model 1`; 2 VMs
  (one per rule, 3 runs each). Training cap 60 min.
- R2b: rule 110, T1, layers 4, seeds 0, 1, 2, `--key-mode v2 --selectors stiff,voi,voiu,conf,rand --save-model 1`;
  3 VMs (one run each). Training cap 50 min.
- 5 on-demand L4s, REMOTE_MINUTES 110 (timeout 122 min); reserve 5 x $2.29 = $11.5 ceiling, expected ~$4-5.

## Decision rules (gemini's R2-D2 and R2-D3, operationalised; 3-seed means, paired 95% t-intervals, 2 df)
- **R2-D2a (D0 gate at 1 layer)**: per rule, validation P0 KL on S1 <= 0.025 AND >= 0.05 on at least two of F, B1, B2.
- **R2-D2b (stiff-first search wins under a depth bottleneck)**: per rule, on every family F/B1/B2 whose TEST P0 KL
  (3-seed mean) is >= 0.05: under the mix pool AND under the sw pool, P2-Stiff or P3-Conf at M = 16 cuts KL vs P0 by >= 25%
  (3-seed mean of per-seed cuts, with the paired CI of the KL difference excluding 0) AND beats P4-Rand under the same pool
  at M = 8 or M = 16 (paired CI of KL(P4) - KL(sel) excluding 0). A family passes if one of {Stiff, Conf} satisfies both
  parts under both pools; the rule passes if every qualifying family passes. If no family qualifies, R2-D2b is "not
  applicable" (no gap), reported as such.
- **R2-D3a (directional key separation)**: rule 110, B1 and B2: 3-seed mean cos(k_fwd, k_bwd) <= 0.35 on each, AND D3a
  redundant-view key cosine >= 0.85 AND D3a drift ratio at m = 5 < 0.20.
- **R2-D3b (uncertainty-gated VoI on bridges)**: rule 110, on each of B1 and B2, under mix AND under sw at M = 8:
  voiu cuts KL vs voi (same models) by >= 30% and vs P0 by >= 35% (3-seed mean cuts; paired CIs of the KL differences
  excluding 0). Also reported: voiu vs round-1's voi numbers (different models; unpaired, descriptive only).
Results go to RESULTS-R2.md (analysis: `analyze_r2.py`), audited in the same way as round 1 if time allows.
