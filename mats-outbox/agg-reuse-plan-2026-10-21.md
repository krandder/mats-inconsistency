# Stage 1 reuse plan: use what is known first (opus-mats, 2026-10-21)

Kelvin (verbatim): "It seems there is quite a lot that is not new, so we should make sure we learn as much as possible from this. Having more that is new is not good, the idea is to use as much of what is already known as possible."

Rule from now on: for every component, implement the established method first, at its published form, and measure it against exact Bayes in aggbench. New designs are justified only by a measured gap that the known methods leave. Every design note names the prior work it extends and says what it changes. Sources: outbox/agg-literature-map-2026-10-21.md.

| Component | Known method to adopt first | What we measure |
|---|---|---|
| Training recipe | Prior-data fitted networks (Müller et al. 2022, arXiv:2112.10510): sample information structures from a prior (the aggbench zoo), train a transformer to output the posterior | Regret to Bayes on held-out structures; this is also Kelvin's full-context transformer bar |
| Set aggregation architecture | Set Transformer SAB+PMA (Lee et al. 2019, arXiv:1810.00825); Deep Sets sum (arXiv:1703.06114) | Same, at matched compute |
| Overlap model (Gaussian) | Winkler GLS pooling (doi:10.1287/mnsc.27.4.479); Partial Information Framework (Satopää et al., arXiv:1505.06472) | Closed-form oracle and fitted-covariance baselines |
| Unknown correlation | Covariance intersection (Julier & Uhlmann 1997) and ICI (doi:10.1016/j.automatica.2017.01.019) | Conservative baseline when provenance is hidden |
| Learned messages between agents | TarMAC (Das et al., arXiv:1810.11187): signature/value/query messages | First time scored against exact Bayes |
| Exact inference + learning | Neural enhanced BP (arXiv:2003.01998); BPNN-D fixed-point-preserving (arXiv:2007.00295); factor-graph tokens for higher-order factors (FGNN, arXiv:1906.00554) | Whether XOR-like structures become learnable |
| Learning structure across tasks | Amortized structure learning: AVICI (arXiv:2205.12934), CSIvA (arXiv:2204.04875) | Relation-classification accuracy vs the pair-classifier probe |
| Second-order signals | Meta-predictions / surprisingly popular (Prelec et al. 2017, doi:10.1038/nature21054); track records via online weights (Hanson LMSR; exponential weights) | Which signal restores Bayes when provenance is hidden |
| Limits (expected failures) | Arieli et al. 2018; Neyman & Roughgarden 2021 (forecasts-only impossibility); Genest & Zidek (convex pools); Hahn 2020 (parity) | Use as predicted ceilings, not as surprises |

Order of work: (1) add the closed-form and learned baselines above to aggbench (opus-mats, measurement side); (2) gemini's model-side runs start from these designs and state their delta; (3) the leaderboard shows the known methods first.
