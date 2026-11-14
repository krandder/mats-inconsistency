## Addendum (29 October, 20:28-21:45Z): M3, M4 and M5 on the existing nets

Run after the coordinator approved them. Pre-registered in `work/scale-free-theory/PREREG-SF.md` (hashed at 20:28Z,
before any outcome; plumbing checks disclosed there). Nets copied to `work/scale-free-theory/nets/` with a hash
manifest. Phase B's registered interventions were not repeated: E8.2's layer-2 query-row patches on the noisy nets,
E22's uniform within-source attention, and E24/E25's matched-pair transplants. A GPT-6 check of code, numbers and
wording (`review3-gpt6-2026-10-29.md`) found no implementation bug, confirmed every count and range, and asked for the
narrower interpretations used below. Learned set: the 9 clean 2-layer nets that learned the task (w32_h2_L2 s0-s4,
w32_h2_L2_mix4 s0, s1, s2, s4); the stuck seed mix4_s3 is reported separately. Numbers are **MEASURED**; readings of
mechanism are marked separately.

**Scorecard.**

| Prediction (PREREG-SF) | Result | Verdict |
|---|---|---|
| P-M3a: K_fit < K_aligned < K_misaligned in both layers, >= 7 of 9 | 7 of 9 | holds |
| P-M3b: routing cost (misaligned minus aligned) larger in layer 1 than in layer 2, >= 7 of 9; weighting cost (aligned minus fit) larger in layer 2, >= 7 of 9 | 4 of 9; 6 of 9 | fails |
| P-M3c: label-plus-own-weight fit loses <= 1/4 of what an aligned transplant loses, both layers, >= 7 of 9 | 4 of 9 (layer 2: 9 of 9; layer 1: 4 of 9) | fails |
| P-M4a: smallest k with KL(net \|\| bottleneck) <= 0.01 is <= 4, >= 7 of 9 | 7 of 9 (k = 3 in three nets, 4 in four, 6 in two) | holds |
| P-M4b: removing context-token variation (k = 0) costs >= 0.05, all 9 | 9 of 9 (0.32-0.82 nats) | holds |
| P-M5a: best head's r(log self-attention, log Bayes variance) >= 0.5, >= 5 of 9 clean nets | 7 of 9 | holds |
| P-M5b: the same, >= 6 of 12 noisy phase-B nets | 8 of 12 | holds |

**M3: sensitivity of each layer to its attention.** KL(net || net with one layer's attention substituted), nats,
ranges over the 9 learned nets; the nets' own KL to Bayes is 0.006-0.022.

| Layer | fit: label table + the episode's own per-key effects | another episode's attention, labels aligned | the same, labels scrambled |
|---|---|---|---|
| 1 | 0.04-0.34 | 0.20-1.05 | 0.61-4.37 |
| 2 | 0.02-0.04 | 0.21-0.41 | 0.44-4.45 |

- Both layers are very sensitive to label alignment: scrambling which token is which costs 0.4-4.4 nats, 20 to several
  hundred times the nets' distance from Bayes. The registered routing contrast (scrambled minus aligned) is positive
  in both layers for 8 of 9 nets (negative in layer 1 of s0).
- Both layers are also sensitive to episode-specific attention: keeping the labels but taking another episode's
  attention costs 0.2-1.05 nats. Part of this may be dependence on the random per-episode IDs rather than on the
  content, so this measures episode-specific attention, not content weighting alone. Even so, these nets' attention
  does more than route by labels.
- In layer 2, the fitted surrogate (label table plus per-key effects) meets the registered relative criterion in 9 of
  9 nets (0.02-0.04 nats, a tenth to a fifth of an aligned transplant). In absolute terms it moves the nets away from
  Bayes (KL(Bayes || fit) 0.026-0.063 against 0.006-0.022). That the per-key effects are precisions, or that layer 2 is
  a Gaussian per-source merge, is not shown by this test (**CONJECTURE**, supported by N7's correlation of the query's
  weights with log precision).
- In layer 1 the surrogate fails the criterion in 5 of 9 nets (0.04-0.34 nats): this particular per-key surrogate does
  not capture block 1's attention, whose episode-specific part seems pair-specific. That fits phase B's E22-E25, which
  locate the big-miss departure in block 1's within-source pooling, but does not identify block 1's computation.
- The layer-role prediction from N7 failed (P-M3b): N7's variance reading ("layer 1 routes, layer 2 weights") is not
  supported causally. In layer 2 the per-key share included label-determined token-type effects (query readings vs
  labelled tokens).
- The stuck seed is sensitive to labels but barely to episode-specific attention (aligned transplants 0.05 and 0.06
  nats in layers 1 and 2; scrambled 1.4 and 0.37), consistent with a net that never learned reliability.

**M4: compression of what block 2 reads from the context tokens.** After block 1, the residual streams of the 15
context tokens (32 dimensions each) were passed through a shared rank-k linear map trained by distillation, with the
query token's state left untouched; that state already attended to the context in block 1, so k = 0 removes the
context tokens' variation, not all context information. Removing it costs 0.32-0.82 nats. A shared rank of 3-4 per
context token sufficed (KL to the net <= 0.01) in 7 of 9 nets (6 in the other two), given the untouched query state; at
k = 4 the bottlenecked nets are about as close to Bayes as the nets (0.009-0.034 against 0.006-0.022 nats). This is an
achieved compression, an upper bound for linear codes shared by both token types, not an intrinsic dimension, not a
per-source message size (a source has five tokens; its Bayes sufficient statistic is two numbers), and not evidence
about the whole cut, since the query's own block-1 channel was not bottlenecked. Distillation may stop at a local
optimum (in one net the k = 2 fit stayed at 0.20 while k = 3 reached 0.009). Outputs: `m4_bottleneck.json` plus the
shards `m4_bottleneck_mix4.json`, `m4_bottleneck_mix4_s2.json`, `m4_bottleneck_mix4_s4.json`.

**M5: self-attention and posterior variance.** In the combining layer, the log of the query token's attention to itself
correlates with the log of Bayes's posterior variance: best-head r 0.56-0.85 in 7 of 9 learned clean nets (s0: 0.27,
s3: 0.43) and 0.56-0.81 in 8 of 12 noisy phase-B nets. Pre-registered exploratory measures: the same correlation with
the nets' own predictive variance is about as high; slopes in log-log are 0.26-0.62, not the 1 an exact sink would give;
one-layer nets reach r 0.16-0.23; the stuck seed -0.02. Post hoc, not pre-registered: all six group-A noisy nets have
r 0.70-0.81, group B 0.34-0.72 (two of six above 0.5). That the query token acts as the prior sink of the
Gaussian-merge identity is a **CONJECTURE** these correlations are consistent with, not something they establish: they
do not show prior-valued values, precision logits or causal use, and Bayes's predictive here is a mixture, for which
the single-Gaussian identity is only approximate.

**What the addendum changes in the report.**
- The routing-plus-weighting decomposition (6.2) gives a surrogate that meets the registered criterion for the
  combining layer but not for block 1; whether layer 2 is a per-source precision-weighted merge with a sink remains a
  conjecture with correlational support (N7, M5).
- "Dense attention mostly pays for routing" is refined: these nets' attention is highly sensitive both to label
  alignment and to episode-specific attention, in both layers.
- What block 2 reads from each context token can be compressed linearly to 3-4 dimensions with little loss (given the
  untouched query state); M4 in 6.3 has a first, partial answer for this cut.
