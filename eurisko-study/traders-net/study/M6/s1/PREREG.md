# M6 stress worker s1: multi-epoch memorisation regime and an independent-evidence family

Written 2026-09-28 ~09:45Z before any run. Claim: `../STATUS.md` 09:36:41Z (s1 = stress items 1 and 2 of 4).
Code: `job/run_s1.py` (imports w1's `m6layer.py` unchanged, copied into `job/`), `summarize.py`.

## Arms (the base-pass survivors plus the kill controls, all in ONE harness = w1's layer, d=64, 16 heads)
- **A** plain logit sum (reference); **B** l0 + Σd (anchored, survived base pass).
- **CG** convex pool, k = softmax(θ), θ [16] global; **CQ** same with θ [Q,16]. Trader belief σ(l0 + d_i);
  pool logit = logit(Σ k_i σ(l0+d_i)) (w2's (c), re-implemented as a `combine` on w1's layer; θ lr 1e-2 as w2).
- **FB01 / FB05** bounded bets: z = l0 + Σ b·tanh(d_i/b), b = 0.1 / 0.5 (w3's (f)).
- Kill controls: **WD** (A + AdamW wd 0.1), **BOOST** (l0 + 0.1·Σd), **ES** (A at the val-chosen epoch).
- Not run: (d)/(e) and Cs (dropped in the base pass); 1-pass/2-pass/FP (w4: identical to each other, fixed point
  not kept; B is the 1-pass layer's nearest analogue here). Committed heads Cm/Cw are w5's harness, not rerun.

## Regimes
- **MS** (item 1, memorisation stress): S0 streams Q=64, T=24032, heldout 0, no regime switch (w1's stream).
  Train positions [0, 2400) only, val [16800,19200), test [19200,24000), `assert_disjoint`; 100 epochs, batch 128,
  eval every 5 epochs. Scored at the final epoch; ES = val-chosen eval point. (w1's ME had 16800 train positions and
  20 epochs and the plain net memorised only 0.011 nats; MS has 7x fewer positions and 5x more epochs.)
- **IE** (item 2, independent evidence): w1's ME split exactly (train [0,16800), 20 epochs, batch 128, eval every
  epoch) on the same stream plus 8 extra questions, family `indep`: y_t ~ Bern(σ(L_t)),
  L_t = (2y_{t-1}−1) + Σ_{j=1..3} (2x_{a_j,t-1}−1), a_j three distinct PRNG questions of the same stream, generated
  with seed-derived RNG. The market l0 (per-question logistic on the question's own window) can see only the own-lag
  term; the three cross-question terms are independent evidence only the trunk sees. Bayes BCE is computed from the
  true p_t on the same test positions.

## Metrics (test split; mean ± sample sd over seeds 1, 2, 3; paired differences for arm comparisons)
- Primary 1, **learnable BCE**: macro mean over periodic, lfsr16, ca110, prime, and, or (IE excluded).
- Primary 2, **humility**: max over {prng, ca30} of (BCE − ln 2).
- Primary 3 (IE only), **IE regret**: BCE on `indep` − Bayes BCE.
- Secondary: AND/OR violation, lfsr16 learnt, ES epoch, train−test gap, pool k effective number.

## Stop rule (the plan's M6 rule, applied per stress regime)
- BCE clause: arm ≥ A × 1.01 learnable BCE in 3/3 seeds.
- Protection clause: arm humility < min(ES, WD, BOOST) humility in 3/3 paired seeds (and, in IE, also IE regret
  ≤ min kill-control IE regret in 3/3 seeds counts as protection on independent evidence).
- An arm is **dropped under stress** if the BCE clause holds and there is no protection. `stress_survivors` = arms not dropped
  in either regime. Pool "beats the kill controls" only if better than min(ES,WD,BOOST) on learnable BCE AND humility in 3/3.
- Predictions (written now): MS: A memorises (humility > 0.03), BOOST and ES stay humble, CG/CQ/FB01 humble but no better
  than BOOST. IE: FB01 cannot express the evidence (16 × 0.1 = 1.6 nats cap vs up to ±3 needed): IE regret ≥ 2x A's;
  CG/CQ not blocked (each head reads the whole residual) but slower.
- Compute: one G1 GPU job (L4 VM), 48 runs, ≤2.5 h reserved (≈ $2.5 of M6's $7; base pass used $0.32). No CPU jobs on this box
  beyond a 1-run smoke test.
