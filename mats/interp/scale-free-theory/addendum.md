## Addendum (29 October, 20:28-21:30Z): M3, M4 and M5 on the existing nets

Run after the coordinator approved them. Pre-registered in `work/scale-free-theory/PREREG-SF.md` (hashed at 20:28Z,
before any outcome; plumbing checks disclosed there). Nets copied to `work/scale-free-theory/nets/` with a hash
manifest. Phase B's registered interventions were not repeated: E8.2's layer-2 query-row patches on the noisy nets,
E22's uniform within-source attention, and E24/E25's matched-pair transplants. GPT-6 checked code, numbers and wording
before this was reported (`review3-gpt6-2026-10-29.md`). Learned set: the 9 clean 2-layer nets that learned the task
(w32_h2_L2 s0-s4, w32_h2_L2_mix4 s0, s1, s2, s4); the stuck seed mix4_s3 is reported separately. All numbers are
**MEASURED**.

**Scorecard.**

| Prediction (PREREG-SF) | Result | Verdict |
|---|---|---|
| P-M3a: K_fit < K_aligned < K_misaligned in both layers, >= 7 of 9 | 7 of 9 | holds |
| P-M3b: routing cost larger in layer 1 than in layer 2, >= 7 of 9; weighting cost larger in layer 2, >= 7 of 9 | 4 of 9; 6 of 9 | fails |
| P-M3c: label-plus-own-weight fit loses <= 1/4 of what an aligned transplant loses, both layers, >= 7 of 9 | 4 of 9 (layer 2: 9 of 9; layer 1: 4 of 9) | fails |
| P-M4a: smallest k with KL(net \|\| bottleneck) <= 0.01 is <= 4, >= 7 of 9 | 7 of 9 (k = 3 in three nets, 4 in four, 6 in two) | holds |
| P-M4b: removing the context message (k = 0) costs >= 0.05, all 9 | 9 of 9 (0.32-0.82 nats) | holds |
| P-M5a: best head's r(log self-attention, log Bayes variance) >= 0.5, >= 5 of 9 clean nets | 7 of 9 | holds |
| P-M5b: the same, >= 6 of 12 noisy phase-B nets | 8 of 12 | holds |

**M3: what attention carries.** KL(net || net with one layer's attention substituted), nats, ranges over the 9 learned
nets; the nets' own KL to Bayes is 0.006-0.022.

| Layer | label routing + the episode's own per-key weights (fit) | another episode's attention, labels aligned | the same, labels scrambled |
|---|---|---|---|
| 1 | 0.04-0.34 | 0.20-1.05 | 0.61-4.37 |
| 2 | 0.02-0.04 | 0.21-0.41 | 0.44-4.45 |

- Routing by labels is essential in both layers: scrambling which token is which costs 0.4-4.4 nats, 20 to several
  hundred times the nets' distance from Bayes.
- Content-dependent weighting is also essential in both layers: keeping the labels but taking another episode's
  weights costs 0.2-1.05 nats. In these nets attention routes *and* weights; "dense attention mostly pays for routing"
  is half the story.
- In the combining layer (2) the decomposition logits = R(labels) + w_j captures almost all of it (0.02-0.04 nats,
  a tenth to a fifth of an aligned transplant): a label-routed, per-source weighted merge, as the Gaussian-merge
  identity (6.1) describes.
- In layer 1 it does not (0.04-0.34 nats; more than a quarter of the aligned cost in 5 of 9 nets): the within-source
  reliability computation uses pair-specific, content-dependent attention, not just a per-key weight. This agrees with
  phase B's E22-E25, which locate the big-miss departure in block 1's within-source pooling.
- Correction to 6.4: N7's variance reading ("layer 1 routes, layer 2 weights") is not supported causally (P-M3b
  fails). In layer 2 the per-key share includes label-determined token-type effects (query readings vs labelled
  tokens), which are routing; in layer 1 the small, episode-specific part of the logits matters a great deal.
- The stuck seed routes but hardly weights: its aligned transplants cost only 0.05 and 0.06 nats (layers 1, 2), its
  scrambled ones 1.4 and 0.37; consistent with a net that never learned reliability.

**M4: internal message size.** After block 1, the 15 context tokens' residual streams (all that block 2 reads from
the context; 32 dimensions each) were passed through a shared rank-k linear map trained by distillation. Removing the
context message (k = 0) costs 0.32-0.82 nats. A rank of 3-4 per token reproduces the net to KL <= 0.01 in 7 of 9 nets
(6 in the other two); at k = 4 the bottlenecked net is about as close to Bayes as the net itself (0.009-0.034 against
0.006-0.022 nats). So what block 2 reads from each context token is about 3-4 dimensions. This is a shared per-token
rank, an upper bound for linear compression across both token types, not a per-source message size (a source has five
tokens; its Bayes sufficient statistic is two numbers); nonlinear or per-type codes could be smaller, and distillation
may stop at a local optimum (one net needed k = 3 after a poor k = 2 fit). It is consistent with the nets passing about
what the task needs between the two rounds, with no sign of a wide tangled channel at this cut.

**M5: the prior as a sink.** In the combining layer, the query token's attention to itself tracks Bayes's posterior
variance: best-head r 0.56-0.85 in 7 of 9 learned clean nets (s0: 0.27, s3: 0.43) and 0.56-0.81 in 8 of 12 noisy
phase-B nets; it tracks the nets' own predictive variance equally well. Slopes in log-log are 0.26-0.62, not the 1
that an exact sink would give: the sink weight moves with the posterior variance but compressed, which fits phase B's
observation that layer 2 spreads weight over sources more evenly than Bayes. One-layer nets: r 0.16-0.23; the stuck
seed: -0.02. Post hoc, not pre-registered: all six group-A noisy nets have r 0.70-0.81, group B 0.34-0.72 (two of six
above 0.5).

**What the addendum changes in the report.**
- The decomposition of 6.2 holds for the combining layer (label routing plus per-key, precision-like weights, with a
  sink that tracks posterior variance), and fails as a description of layer 1, whose content dependence is
  pair-specific.
- "Dense attention mostly pays for routing" is refined: in these nets attention pays for routing and for
  content-dependent weighting, and the weighting in block 1 is not reducible to one scalar per key.
- The "prior as sink" identity is used, imperfectly calibrated, by most nets that learned the task (correlational).
- Between the rounds, the nets pass a low-dimensional message (about 3-4 dimensions per context token); M4 in 6.3 now has a first answer for this cut.
