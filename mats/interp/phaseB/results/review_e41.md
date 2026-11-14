Model: GPT-6

Read-only review, checked 2026-10-30 against the supplied files; no Python or training run.

1. HIGH — Causal exclusion exceeds the experiment. Draft update47_draft.md:5,26,67,69 says amplification and A/B types require missing, multi-source structure. One architecture, target and schedule cannot establish necessity; the toy also has much lower KL and a different prediction target. Fix: “This toy under this recipe does not reproduce E30’s final sign-free gain; multi-source merging remains an untested candidate.” Apply this qualification to the plain-words addition.

2. HIGH — Narrow KL spread does not establish absence of solution types. Draft:7,21,69; Update 46 defines A/B by middle-range departure sign, not KL clustering. E41 measures no such diagnostic (toy_fmt.py:44–57). The cited “1.3–5.4× spread between A and B” actually describes within-seed ctrl/mag improvements (Update 46, E30.1). Fix: report “no large final-KL spread within these 12 seeds per arm”; remove the A/B exclusion. “Every toy run” must say “within each arm.”

3. MEDIUM — Evaluation uncertainty is omitted. toy_fmt.py:46 reuses one 20,000-episode evaluation sample across all models/checkpoints; e41_summ.py:27–32 bootstraps training seeds only. At KL approximately 0.0005, the ctrl–mag difference is approximately 0.000059; these intervals do not quantify episode-sampling uncertainty. Fix: label intervals conditional on this evaluation set, acknowledge unmeasured evaluation error, and avoid population-equivalence language such as “level with mag18” (draft:24). Additional paired evaluation would be needed to quantify it.

4. MEDIUM — The proposed representation explanation is mathematically misleading. Draft:68 contrasts functions of |r| with functions of r², although |r| = sqrt(r²). LayerNorm of a learned affine embedding also generally contains sign-dependent cross terms (minimal.py:51–59). Fix: remove this rationale or specify a testable difference in learned representations/optimization.

5. LOW — Averaging wording obscures the estimand. Draft:4,20: 8.8% is reduction from mean family KL, not exactly base-model KL; neg_pc measures averaging base and negation, not negation alone (toy_fmt.py:53–56). Fix those descriptions.

Survive: registered formats, paired initialization/episode prefixes, switch timing and schedules; exact posterior/sign invariance; KL and geometric-cost formulas; per-seed table rounding and arm summaries. Hashes match. P176 holds; P177/P178 fail numerically as registered. R is mathematically defined but loses its rescue interpretation. Finite-budget path dependence survives; mechanistic and no-types conclusions need rewording; necessity claims fail.
