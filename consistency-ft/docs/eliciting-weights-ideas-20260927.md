# Eliciting which answers should carry the most weight in the projection: ideas (2026-09-27)

Question (Kelvin, 27 Sep): when we project the model's 12 answers onto a coherent joint, which answers
should get the most weight, and how can we elicit that from the model so the weighted projection beats
equal weights? This page lists the hypotheses and how each would be tested. The results are in
docs/eliciting-weights-20260927.md.

**What already failed** (docs/confidence-weighting-20260927.md): several signals were used as per-answer weights, and none helped:
- absolute self-reports (a 90% range, 1–5 knowledge);
- sampling and wording dispersion;
- digit logprobs;
- P(YES) token mass.

Only per-form weights helped (−0.006 nats/answer), mostly by down-weighting NOT P and NOT Q.
Failure pattern: the per-answer signals were proxies for extremeness, which the KL projection already uses.
They were also elicited *about one answer in isolation*, so they cannot see the conflict the projection resolves.
So the promising signals should do at least one of these:
- **(i)** see the other answers;
- **(ii)** come from a different computation than a second guess at the same number;
- **(iii)** be learned across pairs rather than introspected.

## Hypotheses

Each item gives the signal, how to elicit it, why it might work where the others failed, the cost, and a leak-free test.

1. **Predicted revision** (Kelvin's). Ask next to the quick answer: "how far would your probability
   move if you thought about this much longer? Change: 0.xx". Weight = 1/(E|Δ| + τ). Under a martingale
   view of beliefs (effort ladder; [Falck et al. 2024](https://arxiv.org/abs/2406.00793)), E|Δ| is the
   model's own forecast of the noise on this answer. Risk: it is another absolute self-report, like
   the range that failed. Cost: 1 short call per answer. Test: cross-fitted τ. Also check whether it predicts
   the measured revision (item 2), that is, whether the model knows its own instability.
2. **Measured revision.** Compare the quick answer (thinking off) with the same prompt answered with
   thinking on (budget 1,024 tokens). Weight = 1/((Δlogit)² + τ). The second answer comes from a genuinely
   different computation, not a resample: System 1 vs System 2, like test-time compute scaling
   ([Snell et al. 2024](https://arxiv.org/abs/2408.03314)). Also test the thought-through answer as the point to
   project. Cost: about 700–900 reasoning tokens per answer ($0.64/M on 32B).
3. **Self-critique in context.** Show the model all 12 of its quick answers together and ask for each one a
   trust score (1–5) and a revised probability. Unlike items 1–2, the model sees the incoherence and can say
   *which side* is wrong: for example, a misread "NOT" or an OR that is lower than its parts. The literature says to
   expect little: [Huang et al. 2023](https://arxiv.org/abs/2310.01798) find that intrinsic self-correction
   rarely helps, while [Dhuliawala et al. 2023](https://arxiv.org/abs/2309.11495) find that verification in separate steps
   does. Weight = E[trust] from the logprobs. The revised set is also a model-made coherentization to compare
   against. Cost: 1 call per pair.
4. **Comparative evidence directness.** List the 12 statements (no answers) and ask for each one how directly
   the model can judge it, from 5 = it knows facts that bear on it to 1 = it must derive or guess, compared
   against the others. The absolute knowledge rating collapsed to "3". A comparative rating in one context
   might separate direct from composed judgments per pair. Composed judgments are the AND-too-high and OR-too-low
   errors of support theory ([Tversky & Koehler 1994](https://doi.org/10.1037/0033-295X.101.4.547)).
   Cost: 1 call per pair.
5. **Unsupervised truth discovery.** Treat the 12 forms as annotators whose reliability depends on the form (and
   on the pair). Estimate the noise variance of each from its residual to the consensus, reweight, and iterate.
   This follows Dawid–Skene ([1979](https://doi.org/10.2307/2346806)) and CRH-style truth discovery
   ([Li et al. 2016](https://doi.org/10.1145/2897350.2897352)). The per-answer variant is a robust (Huber)
   projection that down-weights outliers. Coherence weighting of human forecasters works
   ([Karvetski et al. 2013](https://doi.org/10.1287/deca.2013.0279);
   [Predd et al. 2008](https://doi.org/10.1287/deca.1080.0119)). Cost: CPU only, no labels.
6. **Supervised stacking of the weight function.** Fit w = exp(θᵀx) by minimizing post-projection log loss on
   the training folds only ([Wolpert 1992](https://doi.org/10.1016/S0893-6080(05)80023-1)). The features x:
   - the form;
   - |logit p|;
   - sampling, wording and digit variance;
   - the residual to the equal-weight fit;
   - any signal from items 1–4.

   This is the upper bound for what the cheap features can do *together*. Cost: CPU only.
7. **Betting against the consensus.** Show the coherent projection's value as a market price and ask how much of
   $100 the model would stake on its own answer: "the market says 0.35, you said 0.20". The stake is a
   Kelly-style conviction measure ([Kelly 1956](https://doi.org/10.1002/j.1538-7305.1956.tb03809.x)) that is
   conditioned on the specific disagreement. Cost: 12 calls per pair. Not tested; it overlaps with item 3.
8. **Base rate + update decomposition.** Elicit the reference-class base rate and the evidence-driven update
   separately, and weight answers by the strength of the evidence they cite. An answer that is only a base
   rate carries little pair-specific information. Cost: longer outputs. Not tested.
9. **Knowledge prerequisites.** Ask checkable background facts about the entities in P and in Q. Weight the answers
   that depend on P (or Q) by quiz accuracy ([Yin et al. 2023](https://arxiv.org/abs/2305.18153) on known
   unknowns). This reweights P-side against Q-side answers per pair, which no per-form weight can do. Needs
   a source of truth for the facts. Not tested.
10. **Meta-prediction ("surprisingly popular").** Ask what probability most other forecasters would give. Where
    the model's own answer departs from its predicted consensus, it may hold private information
    ([Prelec 2004](https://doi.org/10.1126/science.1102081);
    [Prelec et al. 2017](https://doi.org/10.1038/nature21054)). Cost: 12 calls per pair. Not tested.
11. **Framing preference.** Offer 2–3 framings of the same statement (plain, complement, conditional) and ask
    which one the model judges most reliably. Weight by the chosen framing. Per-form data say that NOT framings
    are noisier, and the model may know this per pair. Not tested; the per-form part is already covered.
12. **Conditional anchors.** Elicit P(Q|P) and P(Q|¬P) directly, and weight the joint forms by their agreement
    with the conditionals. This is a local check of consistency with another route to the same joint
    ([Paleka et al. 2024](https://arxiv.org/abs/2412.18544) on consistency checks for LLM forecasters).
    Cost: 2 calls per pair. Not tested.

## What was tested

These were tested on Qwen3-32B, question only, on the 300-pair subset of the 1,066-pair test (the same pairs as
confidence-weighting):
- items 1–4 (paid calls);
- items 5–6 (CPU only).

The structured pairs and Qwen3-235B were run for the best signals.

For every weighted arm:
- τ, or θ, is cross-fitted over 5 folds by pair;
- CIs are paired bootstrap 95% CIs against equal weights and against the per-form weights;
- per-answer error prediction is reported as a Spearman correlation, pooled and partial on |logit p|.

Other general references: [Kadavath et al. 2022](https://arxiv.org/abs/2207.05221);
[Tian et al. 2023](https://arxiv.org/abs/2305.14975); [Xiong et al. 2023](https://arxiv.org/abs/2306.13063);
[Fluri et al. 2023](https://arxiv.org/abs/2306.09983).
