### 6.4 First measurements: interaction profiles (52 nets) and routing classes (15 nets)

**M1 on the clean tiny-PFN cohort** (N5: `work/tiny-pfn/runs`, 8 configurations x 5 seeds, 4,000 episodes x 4 variants
per cut; log-odds of y* + {-0.6, -0.3, 0.3, 0.6} against y*; `n5_interaction.py`, `n5_interaction_v2.json`):

| Nets | cross-source share | deviation share (error), cross-source | within-source share | KL to Bayes |
|---|---|---|---|---|
| exact Bayes | 0 (3.6e-31) | - | 0.039 | 0 |
| Bayes projected to a Gaussian (reference) | 0.054 | - | 0.054 | 0.010 |
| 2 layers, mixture head, learned the task (12) | 0.012-0.041 | 0.34-0.47 | 0.037-0.045 | 0.007-0.032 |
| 2 layers, Gaussian head (15) | 0.054-0.087 | 0.36-0.47 | 0.050-0.062 | 0.016-0.050 |
| 2 layers, mixture head, stuck far from Bayes (3) | 0.109-0.114 | 0.33-0.35 | 0.018 | 0.144-0.145 |
| 1 layer, Gaussian head (10) | 0.186-0.206 | 0.37-0.48 | 0.020-0.024 | 0.174-0.197 |

**M1 on the 12 noisy phase-B nets** (N6, pre-registered in `PREREG-N6.md` before running; noise-marginalised
predictive over 8 draws; `n6_interaction_noisy.py`, `n6_interaction_noisy_v2.json`): group A cross-source share
0.012-0.014, group B 0.015-0.021 (all 6 A nets below the B median); cross-source below within-source in 12 of 12;
correlation with KL 0.95. The three pre-registered predictions hold as stated. But the deviation shares are
0.31-0.33 (A) and 0.28-0.34 (B): the same.

**MEASURED, and what it means.**
- The raw cross-source share tracks KL (0.96 over the 40 clean nets) because the error itself is about 30-48%
  cross-source interaction in every net, good or bad, clean or noisy: the raw share mostly measures the size of the
  error. This confirms the reviewers' smoothness objection; the A/B difference in raw share is B being further from
  Bayes, not a different way of combining.
- What is informative: (1) every net's departure from Bayes is substantially non-additive across sources, so no
  per-source account can describe more than about two thirds of it at this cut (cross-source coupling can enter
  through the normalisation of attention over sources even when the mistake starts within a source); (2) one-layer
  nets and stuck seeds have *less* within-source interaction than Bayes (0.018-0.024 vs 0.039), the signature of not
  applying the per-source nonlinearity (reliability from the sum of squared errors) before mixing, which is what a
  single pool-then-transform round cannot do; (3) post hoc, not pre-registered: B's error is more interactive
  *within* a source than A's (deviation share 0.23-0.24 vs 0.19-0.21, no overlap), in line with phase B's finding
  that B's encoding depends on how a source's errors are spread.

**M2 on 15 clean nets** (N7: `n7_routing_classes.py`, `.json`; w32_h2_L2 and w32_h2_L2_mix4 x 5 seeds, w32_h2_L1 x 5;
1,500 episodes; 16 tokens, fixed layout):

| Layer | routing classes C among 15 rows (TV <= 0.1) | per-key weighting share of row-centred log-attention | label R^2 of the pair-specific part | query mass vs source log precision, r |
|---|---|---|---|---|
| 2-layer nets, layer 1 (20 heads) | 2.7-8.1 | 0.02-0.31 | 0.75-0.89 (median 0.77) | - |
| 2-layer nets, layer 2 (18 heads, without the stuck seed) | 3.1-5.6 | 0.52-0.96 | median 0.04-0.21 | 0.41-0.95 (median 0.87) |
| stuck seed, layer 2 (2 heads) | 2.0-4.4 | 0.58-0.81 | 0.07-0.17 | -0.04 to 0.32 |
| 1-layer nets (10 heads) | 3.7-5.2 | 0.67-0.79 | median 0.50 | 0.13-0.21 |

**MEASURED.** In the 2-layer nets, layer 1 is mostly label routing (few classes; labels explain about 77% of the
pair-specific logits) and layer 2 is mostly content weighting (one scalar per key carries most of the variation, and
the query's weights on the sources' readings track their log precision, median r = 0.87): routing, then a
precision-weighted merge, as the decomposition and the sink identity describe. One-layer nets cannot weight by
reliability (r about 0.18), since reliability must be gathered within a source before the query can use it. Caveat:
with the fixed token layout, label routing coincides with positional routing; permuting token order is the next
check (M3).
