# Overnight report, 2026-10-28 (opus-mats; /goal 01:45Z to ~16:00Z)

Goal (Kelvin): find PFN prior worlds where small transformers show in-context (not prior-level) dedup and market-like
aggregation; pilot them; start the mechanistic study of the best one; produce sharp crux tests he and gemini would
accept. Status of every claim below: from the named report files; most results are single runs or few seeds and
were adversarially reviewed only where stated.

## Headlines
1. **Plateau = source-binding failure (astra; decisive on its test).** Tiny-PFN seeds that stall know how reliable the
   sources look collectively but never attach reliability to the right source. Fresh-data intervention (swap which
   source's labelled data sits with which query reading, one 3-cycle): stalled models move 0.2-0.5% of Bayes's change,
   acquired models 82-99%; layer-1 own-source attention ratio ~1 vs 15-233. A 4-page LaTeX paper:
   outbox/binding-plateau-paper-2026-10-28/paper.pdf (Codex review: review-codex.md there).
2. **A working in-context circuit in the market world (W2b).** Models calibrate sources in context (crux C4 passes at
   N=96: 84-94% of matched pairs). Circuit: block 1 compares each quote with the item consensus; blocks 2-3 pool a
   source's other quotes into its reliability; the query token reads quotes by that reliability. Cutting the single
   block-3 edge carrying a source's other quotes into its query-item quote drops C4 closure 94%->28% (random cuts: no
   effect). outbox/w2-market-mechanism-2026-10-28.md (Codex review pending/at -review file).
3. **Two failures replicated across worlds:**
   - Scaling with evidence: plain attention's predictive variance does not shrink with N beyond training (W1: ~140-200x
     Bayes at N=4096); gap to Bayes grows with N in W1 and W2. Minimal fixes failed (unnormalised sum head blows up;
     attention sink no effect; training to N=512 halves the gap but it still grows). outbox/world1-junta-pilot.
   - Shared-evidence discounting: relays / shared roots are treated as independent (W2b C7 ~0); copies are discounted
     mainly through exact value ties, and only when copies are common in training (dedup report: 0.25->0.77 at depth 4).
     outbox/pfn-dedup-depth-2026-10-26.md (rebuilt after review 2).
4. **Inconsistency minimisation (crux C6):** point/MAP minimisation can badly misstate probabilities (W1 example 0.04 vs
   Bayes 0.82); residual fitting cannot recover prior information; a correctly specified free energy recovers Bayes.
   Does NOT refute the program's distribution-level objectives (GPT-6 review corrected my first reading).
   outbox/c6-inconsistency-vs-bayes-2026-10-28.md.
5. **Tiny-model errors from Bayes:** budget = 22% ID-vector crosstalk, 21% residual direction, 57% shared systematic;
   a third layer cuts the systematic part 46% (a second, sharper readout); extra heads don't. ARC bar (beat sampling
   mechanistically) not met. outbox/tiny-pfn-interp-step1-2026-10-27.md.

**Plateau study (astra, 13:48Z, outbox/tiny-pfn-plateau-2026-10-28.md):** acquisition of the in-context solution
by 80k steps: width 16 baseline 39/50, lower Adam epsilon 39/50 (no rescue), larger batch 44/50 at equal exposure
(paired p = 0.0625); width 32: 47/50, 46/50, 48/50. "Query-only" is inaccurate for stalled models: they use some
non-query context but not source association. Optimisation mechanism and remedy not established. Cost ~$6.

## World selection (3 critique rounds, 4 model families; then Kelvin's redirect)
- DESIGN-v1 (work/world-critique/): W1 similarity (hierarchical junta), W2 market (biased sources, provenance),
  W3 chains (discrete), cruxes C1-C7. Literature sweep (17 agents + verifier): outbox/lit-aggregation-priors-2026-10-28.md;
  novelty check: outbox/lit-novelty-check-2026-10-28.md (in-context dedup vs exact Bayes, policy invariance at matched
  sufficient statistics, Bayes-correctness at N >> N_train: still open).
- W3 chains: parked (XOR chains unlearnable in-context; equality variant had a design bug).
- Kelvin, 12:17Z: W2 is measurement aggregation, not what he wants -> microstructure deep dive
  (outbox/microstructure-priors-2026-10-28.md; critiques work/micro-critique/): beauty contest / Kyle-LOP need study.
- Kelvin, 13:15Z: chosen main PFN setup = **World FA** (forecast aggregation; target = fundamental value; difficulty =
  hidden correlated information structure learned from a history of resolved rounds). Spec work/world-fa/SPEC.md;
  3-model critique and builder running.

## Crux tests (current status)
| Crux | Status |
|---|---|
| C1 policy-invariant aggregation extrapolates (N >> N_train) | fails in W1/W2 (gap grows with N); formal test vacuous (controls never reach KL<0.05) |
| C2 depth vs chain length (looped nets) | not readable (W3 parked) |
| C3 bounded consensus | analytic bound noted (lossless bounded consensus impossible in general); not tested |
| C4 in-context source inference | passes provisionally at N=96 (W2b), collapses at N=512 |
| C5 optimisation vs expressivity | binding plateau result; dense supervision speeds C4 (60k vs 140k steps) without lowering final gap |
| C6 inconsistency minimisation vs Bayes | point estimates fail; free energy fine |
| C7 shared-evidence dedup | fails (W2b and W2c: dedup fraction -0.23..0.19 even with relay-rich training; provenance ids unused). Caveat: C7 queries need two-hop DIFF paths the models don't follow (0.5-0.9 nats off Bayes there), so a one-hop dedup test is needed — to be built into World FA, not W2. W2c also lifted C4 at N=512 to 76-86% (from 17-29%). |

## Started for the next phase (Kelvin, 13:15Z)
- **Change-of-measure interpretability program** (workflow running): Gaussianise layer l conditional on known latents,
  density ratio w = p/q by classifier, tempering curve L(alpha), refine q until L(0)~L(1) under a complexity budget
  and cross-layer consistency. Phase A: design panel -> METHOD.md -> validated toolkit -> round 1 on tiny PFN -> 2
  adversarial reviews (work/cm-interp/). astra to audit after its plateau report.
- **World FA** build (work/world-fa/).

## Spend
GPU ~ $80-90 of the $200 (estimability 12.2, dedup 15.2, tiny-PFN 5.7, W1 14.4, W2 11.0, W3 8.5, W2c <=15 running,
astra plateau <=10 running). All finished VMs verified deleted by their agents.

## Corrections made overnight
- "Depth 1 is prior-level only" (dedup) — wrong; confounded test (review 2). Rebuilt.
- "Our atom energy F gives 0.5 vs Bayes 0.8" (C6) — wrong; misspecified inputs (GPT-6 review). Withdrawn.
- Searchformer "6.2M vs 15M params" premise — mixed trainable/total counts (astra). Lane paused awaiting Kelvin.
- Astra's plateau assignment sat unread ~9 h (no tmux doorbell from me); fixed 10:46Z.
