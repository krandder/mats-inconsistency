Checked 2026-10-31 04:55 UTC against the saved code, logs and JSON; no files modified. References: **R** = [report](/home/agent/workspaces/mats/outbox/tractable-gates-2026-10-31.md), **T** = [theory v2.1](/home/agent/workspaces/mats/work/tractable-gates/notes/theory.md).

First-round dispositions, using the original numbering:

1. **Partly** — R:30–36,161–166 adds scale conditions; T:62–68 still overstates the unbounded-scale exponent.
2. **Partly** — R:365–366 distinguishes P12 and leaves trained ranks open; R:25–27,363 retains broader conclusions.
3. **Yes** — R:90–94,189–191 withdraws the unsupported 10M+ conclusion.
4. **Yes** — R:203–206; T §8 correctly labels the sorting construction sufficient.
5. **Partly** — R:243–262 adds ReLU’s alternative implementation; error matching and consistent costing remain incomplete.
6. **Partly** — R:342–348 reports numerical sensitivity, but quadrature and Mehler orders change together.
7. **Partly** — R:130–132,233–234 discloses splicing; the summary still presents phase as decisive.
8. **Partly** — R:214–224 shows held-out ranks; R:48–54 still overstates required summary sizes.
9. **Yes** — R:280–292 discloses coupling and supplies four independently clipped arms; original trained-net results remain coupled.
10. **Partly** — R:380 fixes P9’s softplus scores; R:381 still uses the wrong PFN aggregation.
11. **Yes** — R:407–408 and `logs/hermite_classes.log` replace the defective tails and mark unresolved degrees; results remain uncertified.
12. **Partly** — R:167–188 specifies the ray law and numerical limitations; summary extrapolations remain.
13. **Partly** — R:239,270–271,340–341 adds the floor, tie and MC wins; uncertainty language remains too categorical.
14. **Yes** — R:110–114 explains rounded budgets, omitted costs and Sobol scrambling.
15. **Yes** — R:193–201; T §5 restricts the radius theorem and withdraws fixed-precision degree growth.
16. **Yes** — R:157–158; T §6 restricts the instability proof to bilinear gates.
17. **Yes** — R:232,409 and `lib/gnet.py` correct biased tanh’s diagnostic to approximately 1.17.
18. **Yes** — R:375,377,402 and score artifacts fix P4 completeness, pooled P6 and rank provenance.
19. **Yes** — R:87,258,361 acknowledges non-monotone exceptions.

Remaining issues:

1. **Major — R:20–27,88,363–364: conclusions still exceed the experiment.** “No deductive advantage” covers more than the tested K2 implementation; trained summary ranks remain unmeasured. “Competitive” gates being nearly linear/constant also mixes random-net diagnostics with trained nets: trained softplus1’s linear variance fraction is only approximately **0.55** (`R_softplus1.json`). **Fix:** state the observed result precisely: zero joint MC/RQMC wins among the original 45 trained-net K2 evaluations; broader estimator and summary tractability remain open.

2. **Major — R:22–24,48–54,227–234: phase and rank claims remain overstated.** Phase was not isolated from splicing, scale or normalization. “Every gate … far fewer” is unsupported by in-sample ranks alone: biased tanh’s held-out ranks are **103/146**, exceeding 100. Sharp softplus16 “matches ReLU” only approximately at width 16; width 8 gives **21 versus 9**. Likewise, §3’s “within 25%” fails there for both softplus16 and smoothstep (**12 versus 9**). **Fix:** describe associations and finite-ensemble estimates, and specify widths.

3. **Major — R:77,380: “every gate’s inputs grow 1.4–1.9x” is false.** Sorting tanh’s median registered statistic is **0.416**, and its measured trained/initial ratio is **0.399**: its inputs shrink. Moreover, dividing by nominal √2 does not measure actual growth from initialization; ReLU gives **1.473 nominal versus 1.283 measured growth**. **Fix:** distinguish the two statistics and include tanh’s exception.

4. **Major — R:381,401: P10 is still not scored literally.** From `runs/pfn/pfn_softplus1_s*/eval_act.json`, compute each layer’s RMS as `sqrt(mean(unit_rms²))`, then the median over layers divided by √2. Seeds 0–2 give **2.720, 2.316, 2.050**, not **1.7–2.2**. The latter is the median over units. **Fix:** use the registered layer statistic; the refuted verdict remains unchanged.

5. **Major — R:57–61,259–262,342–348: the implementation comparison remains incomplete.** The new ReLU ratios/wins reproduce, but equal machinery does not establish equal numerical accuracy. `lib/k2gen.py` still charges gsrelu **154** per evaluation under otherwise different transcendental conventions. Trained convergence changes quadrature and Mehler order simultaneously, preventing attribution or an error-matched cost comparison. **Fix:** retain these as implementation-specific observations; independently vary both orders, select tolerances and rescore budgets before stronger comparisons.

6. **Major — T:62–68: the unbounded-scale theorem retains the bounded-scale exponent.** Section 1 derives **log(1/ε)^{n(1+q/2)/γ}** under its Gaussian-tail repair, whereas Theorem A invokes those same conditions and states **log(1/ε)^{n/γ}**. **Fix:** separate bounded and unbounded cases, or claim only an unspecified polylogarithmic bound for the latter.

7. **Minor — R:40–46: ray claims exceed resolved numerical coverage.** “Every ε≤0.1” includes untested precisions. “Softplus … lose at every tested ε down to 10⁻⁶” includes unresolved entries, notably softplus16/8 at fine precision and softplus4 at width 16, ε=10⁻⁶. **Fix:** say “every resolved tested precision” and identify unresolved comparisons.

8. **Minor — R:413: the splice-scale range omits deeper layers.** **0.5–3.6** describes approximately the first back layer only. `posthoc_review.log` reports final-layer extremes **0.06–8.48**. **Fix:** label the layer and sampled coverage, or report both ranges.

9. **Minor — R:263,270–271: winner and uncertainty terminology needs qualification.** “Only soft ordered gates win” must mean the registered **≥3/4 cell rule**; the new table contains two individual ReLU wins at depth 16. Floor-limited gain estimates are not guaranteed lower bounds without uncertainty analysis. **Fix:** distinguish individual and cell wins; describe reference-noise bias rather than certified bounds.

10. **Minor — R:65–66,285–291: rounding errors.** Raw sorting means give ReLU **0.103461 → 0.103**, clipped GELU **0.099462 → 0.099**, softplus16/ReLU **0.985297 → 0.99**, and gsrelu/ReLU **1.165489 → 1.17**. The reported clipping ratios **0.88/1.08/1.66**, trained counts **0/45 joint, 6 MC**, and pooled P6 **2.202/0.514/0.343** reproduce.

**Verdict: major revision required; substantial repairs succeeded, but headline scope and two sharpening claims remain materially wrong.**
tokens used
105,237
Checked 2026-10-31 04:55 UTC against the saved code, logs and JSON; no files modified. References: **R** = [report](/home/agent/workspaces/mats/outbox/tractable-gates-2026-10-31.md), **T** = [theory v2.1](/home/agent/workspaces/mats/work/tractable-gates/notes/theory.md).

First-round dispositions, using the original numbering:

1. **Partly** — R:30–36,161–166 adds scale conditions; T:62–68 still overstates the unbounded-scale exponent.
2. **Partly** — R:365–366 distinguishes P12 and leaves trained ranks open; R:25–27,363 retains broader conclusions.
3. **Yes** — R:90–94,189–191 withdraws the unsupported 10M+ conclusion.
4. **Yes** — R:203–206; T §8 correctly labels the sorting construction sufficient.
5. **Partly** — R:243–262 adds ReLU’s alternative implementation; error matching and consistent costing remain incomplete.
6. **Partly** — R:342–348 reports numerical sensitivity, but quadrature and Mehler orders change together.
7. **Partly** — R:130–132,233–234 discloses splicing; the summary still presents phase as decisive.
8. **Partly** — R:214–224 shows held-out ranks; R:48–54 still overstates required summary sizes.
9. **Yes** — R:280–292 discloses coupling and supplies four independently clipped arms; original trained-net results remain coupled.
10. **Partly** — R:380 fixes P9’s softplus scores; R:381 still uses the wrong PFN aggregation.
11. **Yes** — R:407–408 and `logs/hermite_classes.log` replace the defective tails and mark unresolved degrees; results remain uncertified.
12. **Partly** — R:167–188 specifies the ray law and numerical limitations; summary extrapolations remain.
13. **Partly** — R:239,270–271,340–341 adds the floor, tie and MC wins; uncertainty language remains too categorical.
14. **Yes** — R:110–114 explains rounded budgets, omitted costs and Sobol scrambling.
15. **Yes** — R:193–201; T §5 restricts the radius theorem and withdraws fixed-precision degree growth.
16. **Yes** — R:157–158; T §6 restricts the instability proof to bilinear gates.
17. **Yes** — R:232,409 and `lib/gnet.py` correct biased tanh’s diagnostic to approximately 1.17.
18. **Yes** — R:375,377,402 and score artifacts fix P4 completeness, pooled P6 and rank provenance.
19. **Yes** — R:87,258,361 acknowledges non-monotone exceptions.

Remaining issues:

1. **Major — R:20–27,88,363–364: conclusions still exceed the experiment.** “No deductive advantage” covers more than the tested K2 implementation; trained summary ranks remain unmeasured. “Competitive” gates being nearly linear/constant also mixes random-net diagnostics with trained nets: trained softplus1’s linear variance fraction is only approximately **0.55** (`R_softplus1.json`). **Fix:** state the observed result precisely: zero joint MC/RQMC wins among the original 45 trained-net K2 evaluations; broader estimator and summary tractability remain open.

2. **Major — R:22–24,48–54,227–234: phase and rank claims remain overstated.** Phase was not isolated from splicing, scale or normalization. “Every gate … far fewer” is unsupported by in-sample ranks alone: biased tanh’s held-out ranks are **103/146**, exceeding 100. Sharp softplus16 “matches ReLU” only approximately at width 16; width 8 gives **21 versus 9**. Likewise, §3’s “within 25%” fails there for both softplus16 and smoothstep (**12 versus 9**). **Fix:** describe associations and finite-ensemble estimates, and specify widths.

3. **Major — R:77,380: “every gate’s inputs grow 1.4–1.9x” is false.** Sorting tanh’s median registered statistic is **0.416**, and its measured trained/initial ratio is **0.399**: its inputs shrink. Moreover, dividing by nominal √2 does not measure actual growth from initialization; ReLU gives **1.473 nominal versus 1.283 measured growth**. **Fix:** distinguish the two statistics and include tanh’s exception.

4. **Major — R:381,401: P10 is still not scored literally.** From `runs/pfn/pfn_softplus1_s*/eval_act.json`, compute each layer’s RMS as `sqrt(mean(unit_rms²))`, then the median over layers divided by √2. Seeds 0–2 give **2.720, 2.316, 2.050**, not **1.7–2.2**. The latter is the median over units. **Fix:** use the registered layer statistic; the refuted verdict remains unchanged.

5. **Major — R:57–61,259–262,342–348: the implementation comparison remains incomplete.** The new ReLU ratios/wins reproduce, but equal machinery does not establish equal numerical accuracy. `lib/k2gen.py` still charges gsrelu **154** per evaluation under otherwise different transcendental conventions. Trained convergence changes quadrature and Mehler order simultaneously, preventing attribution or an error-matched cost comparison. **Fix:** retain these as implementation-specific observations; independently vary both orders, select tolerances and rescore budgets before stronger comparisons.

6. **Major — T:62–68: the unbounded-scale theorem retains the bounded-scale exponent.** Section 1 derives **log(1/ε)^{n(1+q/2)/γ}** under its Gaussian-tail repair, whereas Theorem A invokes those same conditions and states **log(1/ε)^{n/γ}**. **Fix:** separate bounded and unbounded cases, or claim only an unspecified polylogarithmic bound for the latter.

7. **Minor — R:40–46: ray claims exceed resolved numerical coverage.** “Every ε≤0.1” includes untested precisions. “Softplus … lose at every tested ε down to 10⁻⁶” includes unresolved entries, notably softplus16/8 at fine precision and softplus4 at width 16, ε=10⁻⁶. **Fix:** say “every resolved tested precision” and identify unresolved comparisons.

8. **Minor — R:413: the splice-scale range omits deeper layers.** **0.5–3.6** describes approximately the first back layer only. `posthoc_review.log` reports final-layer extremes **0.06–8.48**. **Fix:** label the layer and sampled coverage, or report both ranges.

9. **Minor — R:263,270–271: winner and uncertainty terminology needs qualification.** “Only soft ordered gates win” must mean the registered **≥3/4 cell rule**; the new table contains two individual ReLU wins at depth 16. Floor-limited gain estimates are not guaranteed lower bounds without uncertainty analysis. **Fix:** distinguish individual and cell wins; describe reference-noise bias rather than certified bounds.

10. **Minor — R:65–66,285–291: rounding errors.** Raw sorting means give ReLU **0.103461 → 0.103**, clipped GELU **0.099462 → 0.099**, softplus16/ReLU **0.985297 → 0.99**, and gsrelu/ReLU **1.165489 → 1.17**. The reported clipping ratios **0.88/1.08/1.66**, trained counts **0/45 joint, 6 MC**, and pooled P6 **2.202/0.514/0.343** reproduce.

**Verdict: major revision required; substantial repairs succeeded, but headline scope and two sharpening claims remain materially wrong.**
